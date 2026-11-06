# -*- coding: utf-8 -*-
"""Tests unitaires du Seg-Grad-CAM multi-niveaux (MultiLevelCAM) de xai_matched_kaggle.py.

Le script Kaggle s'execute a l'import (installation, clonage, donnees) : on n'en charge que les
definitions testees, par analyse syntaxique (ast). On teste donc EXACTEMENT le code qui a produit
les resultats de results/phase3_ablation/xai_matched/, sans le modifier.

    .venv\\Scripts\\python.exe -m pytest phase3_ablation/xai_matched/tests -q
"""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT = Path(__file__).resolve().parents[1] / "xai_matched_kaggle.py"
WANTED = {"MultiLevelCAM", "in_organ_ratio", "pointing_accuracy", "spatial_entropy", "boundary_energy_ratio",
          "activation_dice", "centroid_distance_mm", "all_metrics", "ssim_3d", "SPACING", "LAYERS"}
N_CLASSES = 7            # fond + 6 OAR (poumon G/D, coeur, oesophage, moelle, aorte)
SHAPE = (32, 32, 16)     # (H, W, D) divisible par 4 pour le faux U-Net


def _load():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in WANTED:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in node.targets):
            keep.append(node)
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "ndi": ndi}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SCRIPT), "exec"), ns)
    missing = WANTED - set(ns)
    assert not missing, f"definitions absentes du script : {missing}"
    return ns


NS = _load()
MultiLevelCAM, ssim_3d, all_metrics = NS["MultiLevelCAM"], NS["ssim_3d"], NS["all_metrics"]


# ------------------------------------------------------------------ faux U-Net
class MockUNet(nn.Module):
    """U-Net 3D minimal (2 niveaux, skip, InstanceNorm, PReLU), < 20 k parametres, CPU."""

    def __init__(self, n_classes=N_CLASSES, c=8):
        super().__init__()
        self.enc0 = nn.Sequential(nn.Conv3d(1, c, 3, padding=1), nn.InstanceNorm3d(c), nn.PReLU())
        self.enc1 = nn.Sequential(nn.Conv3d(c, 2 * c, 3, stride=2, padding=1), nn.InstanceNorm3d(2 * c), nn.PReLU())
        self.bottom = nn.Sequential(nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=1), nn.InstanceNorm3d(4 * c),
                                    nn.PReLU())
        self.up1 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(4 * c, 2 * c, 3, padding=1), nn.InstanceNorm3d(2 * c), nn.PReLU())
        self.up0 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv3d(2 * c, c, 3, padding=1), nn.InstanceNorm3d(c), nn.PReLU())
        self.head = nn.Conv3d(c, n_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        b = self.bottom(e1)
        d1 = self.dec1(torch.cat([self.up1(b), e1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], 1))
        return self.head(d0)


MOCK_LAYERS = {"enc_profond": "bottom", "dec_demi": "dec1", "dec_pre_tete": "dec0", "sortie": "head"}


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def model():
    torch.manual_seed(0)
    return MockUNet().eval()


@pytest.fixture
def cam(model):
    c = MultiLevelCAM(model, MOCK_LAYERS)
    yield c
    c.close()


def _make_volume(seed=1):
    """Volume (1, 1, H, W, D) : fond bruite + deux ellipsoides de contraste different, et leur label."""
    g = torch.Generator().manual_seed(seed)
    x = 0.1 * torch.rand(1, 1, *SHAPE, generator=g)
    zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in SHAPE], indexing="ij")
    big = (zz / 0.5) ** 2 + (yy / 0.3) ** 2 + (xx / 0.6) ** 2 < 1
    small = ((zz + 0.5) / 0.2) ** 2 + ((yy - 0.5) / 0.2) ** 2 + (xx / 0.3) ** 2 < 1
    x[0, 0][big] += 0.8
    x[0, 0][small] += 0.4
    y = torch.zeros(1, *SHAPE, dtype=torch.long)
    y[0][big], y[0][small] = 1, 2
    return x, y


@pytest.fixture
def volume():
    return _make_volume()[0]


@pytest.fixture(scope="module")
def trained_model():
    """Faux U-Net entraine 80 iterations (CPU, quelques secondes) a segmenter les 2 ellipsoides : ses poids
    sont APPRIS, ce qui donne un sens au test d'Adebayo (appris vs aleatoire)."""
    torch.manual_seed(0)
    m = MockUNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    data = [_make_volume(s) for s in range(1, 5)]
    for it in range(80):
        x, y = data[it % len(data)]
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    m.eval()
    x, y = data[0]
    with torch.no_grad():
        p = m(x).argmax(1)
    dice = 2 * ((p == 1) & (y == 1)).sum() / ((p == 1).sum() + (y == 1).sum())
    assert dice > 0.8, f"entrainement du faux modele insuffisant (Dice {dice:.2f})"
    return m


@pytest.fixture
def present_classes(model, volume):
    with torch.no_grad():
        pred = model(volume).argmax(1)
    cls = sorted(int(c) for c in pred.unique())
    assert cls, "le faux modele ne predit aucune classe"
    return cls


def _first_level(maps):
    return next(iter(next(iter(maps.values())).values()))


# ------------------------------------------------------------------ 1. mock / construction
def test_mock_est_leger_et_cpu(model):
    assert sum(p.numel() for p in model.parameters()) < 50_000
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_couche_inexistante_refusee(model):
    with pytest.raises(ValueError, match="introuvables"):
        MultiLevelCAM(model, {"enc_profond": "n_existe_pas"})


def test_close_retire_tous_les_hooks(model):
    c = MultiLevelCAM(model, MOCK_LAYERS)
    assert sum(len(m._forward_hooks) for m in model.modules()) == len(set(MOCK_LAYERS.values()))
    c.close()
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0


# ------------------------------------------------------------------ 2. dimensions
def test_dimensions_heatmap_egales_entree(cam, volume, present_classes):
    maps, npred, shapes = cam(volume, present_classes)
    assert set(maps) == set(present_classes)
    for c in present_classes:
        assert set(maps[c]) == set(MOCK_LAYERS)
        for lv, h in maps[c].items():
            assert h.shape == SHAPE, (c, lv, h.shape)
            assert h.dtype == np.float32


def test_formes_des_cartes_de_features(cam, volume, present_classes):
    _, _, shapes = cam(volume, present_classes)
    H, W, D = SHAPE
    assert shapes["enc_profond"] == [32, H // 4, W // 4, D // 4]
    assert shapes["dec_demi"] == [16, H // 2, W // 2, D // 2]
    assert shapes["dec_pre_tete"] == [8, H, W, D]
    assert shapes["sortie"] == [N_CLASSES, H, W, D]


@pytest.mark.parametrize("shape", [(16, 16, 8), (48, 32, 16), (32, 64, 24)])
def test_dimensions_non_cubiques(model, shape):
    x = torch.rand(1, 1, *shape, generator=torch.Generator().manual_seed(2))
    c = MultiLevelCAM(model, MOCK_LAYERS)
    try:
        maps, _, _ = c(x, [0])
    finally:
        c.close()
    assert all(h.shape == shape for h in maps[0].values())


# ------------------------------------------------------------------ 3. bornes / normalisation
def test_bornes_0_1_et_min_nul(cam, volume, present_classes):
    maps, npred, _ = cam(volume, present_classes)
    for c in present_classes:
        for lv, h in maps[c].items():
            assert np.isfinite(h).all(), (c, lv)
            assert h.min() >= 0.0 and h.max() <= 1.0, (c, lv, h.min(), h.max())
            if npred[c] > 0 and h.max() > 0:
                # min-max avec diviseur (max - min + 1e-8) : min = 0 exactement, max = r / (r + 1e-8) ou r est
                # l'etendue brute ; r ~ 1e-4 sur le faux modele -> max ~ 0,9999 (et non 1 exactement)
                assert h.min() == 0.0
                assert h.max() == pytest.approx(1.0, abs=1e-3)


def _reference_cam(model, x, cls, layer):
    """Implementation de reference independante (torch.autograd.grad, sans hook de tenseur)."""
    store = {}
    h = dict(model.named_modules())[layer].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        logits = model(x)
        mask = (logits.argmax(1, keepdim=True) == cls).float()
        (g,) = torch.autograd.grad((logits[:, cls:cls + 1] * mask).sum(), store["a"])
    finally:
        h.remove()
    raw = (g.mean(dim=(2, 3, 4), keepdim=True) * store["a"]).sum(1, keepdim=True)
    cam = F.interpolate(F.relu(raw), size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy(), raw.detach()


@pytest.mark.parametrize("level", list(MOCK_LAYERS))
def test_egal_a_une_implementation_de_reference(cam, model, volume, present_classes, level):
    """Verifie formule complete : poids = moyenne spatiale des gradients, ReLU, interpolation, min-max."""
    cls = present_classes[-1]
    maps, _, _ = cam(volume, [cls])
    ref, raw = _reference_cam(model, volume, cls, MOCK_LAYERS[level])
    np.testing.assert_allclose(maps[cls][level], ref, atol=1e-5)
    if (raw < 0).any():   # la ReLU a bien supprime des contributions negatives
        assert maps[cls][level].min() == 0.0


# ------------------------------------------------------------------ 4. cas limites
def test_volume_nul_reste_fini_et_borne(cam):
    x = torch.zeros(1, 1, *SHAPE)
    maps, npred, _ = cam(x, list(range(N_CLASSES)))
    for c in range(N_CLASSES):
        for h in maps[c].values():
            assert np.isfinite(h).all()
            assert h.min() >= 0.0 and h.max() <= 1.0
    assert sum(npred.values()) == int(np.prod(SHAPE))     # chaque voxel predit dans une seule classe


def test_classe_non_predite_donne_carte_nulle_et_metriques_nan(trained_model, volume):
    with torch.no_grad():
        predites = set(trained_model(volume).argmax(1).unique().tolist())
    absent = [c for c in range(N_CLASSES) if c not in predites]
    assert absent, "le modele entraine (2 ellipsoides) ne devrait pas predire les classes 3-6"
    cam = MultiLevelCAM(trained_model, MOCK_LAYERS)
    try:
        maps, npred, _ = cam(volume, [absent[0]])
    finally:
        cam.close()
    assert npred[absent[0]] == 0
    for h in maps[absent[0]].values():
        assert not h.any()
        m = all_metrics(h, np.ones(SHAPE, bool))
        assert np.isnan(m["in_organ_ratio"]) and np.isnan(m["spatial_entropy"])


def test_nan_en_entree_est_detectable(cam, volume):
    """Comportement documente : le code ne filtre pas les NaN. Une entree contenant des NaN doit produire une
    carte NON finie (detectable en aval), jamais une carte d'apparence valide."""
    x = volume.clone()
    x[0, 0, 0, 0, 0] = float("nan")
    maps, _, _ = cam(x, [0])
    assert any(not np.isfinite(h).all() for h in maps[0].values())


def test_batch_superieur_a_1_explique_le_premier_echantillon(model, volume):
    """Le code explique x[0] (indexation [0, 0]). Avec un modele sans couplage inter-echantillons (InstanceNorm,
    mode eval), ajouter un 2e echantillon ne doit pas changer la carte du 1er."""
    other = torch.flip(volume, dims=[2])
    c = MultiLevelCAM(model, MOCK_LAYERS)
    try:
        ref, _, _ = c(volume, [1])
        bat, _, _ = c(torch.cat([volume, other], 0), [1])
    finally:
        c.close()
    for lv in MOCK_LAYERS:
        np.testing.assert_allclose(bat[1][lv], ref[1][lv], atol=1e-5)
        assert bat[1][lv].shape == SHAPE


def test_determinisme(cam, volume, present_classes):
    a, _, _ = cam(volume, present_classes)
    b, _, _ = cam(volume, present_classes)
    for c in present_classes:
        for lv in MOCK_LAYERS:
            np.testing.assert_array_equal(a[c][lv], b[c][lv])


def test_appels_successifs_independants(cam, volume, present_classes):
    """Les gradients d'un appel ne doivent pas fuir dans le suivant (zero_grad + remise a zero des dicts)."""
    seul, _, _ = cam(volume, [present_classes[0]])
    cam(torch.flip(volume, dims=[3]), present_classes)
    encore, _, _ = cam(volume, [present_classes[0]])
    for lv in MOCK_LAYERS:
        np.testing.assert_array_equal(seul[present_classes[0]][lv], encore[present_classes[0]][lv])


def test_ordre_des_classes_sans_effet(cam, volume, present_classes):
    """retain_graph est utilise pour toutes les classes sauf la derniere : l'ordre ne doit rien changer."""
    a, _, _ = cam(volume, present_classes)
    b, _, _ = cam(volume, present_classes[::-1])
    for c in present_classes:
        for lv in MOCK_LAYERS:
            np.testing.assert_allclose(a[c][lv], b[c][lv], atol=1e-6)


def test_ne_modifie_pas_les_poids(model, volume, present_classes):
    before = copy.deepcopy(model.state_dict())
    c = MultiLevelCAM(model, MOCK_LAYERS)
    try:
        c(volume, present_classes)
    finally:
        c.close()
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


# ------------------------------------------------------------------ 5. test d'Adebayo
def _randomize(module, seed=123):
    g = torch.Generator().manual_seed(seed)
    nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="relu", generator=g)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _spearman(a, b):
    """Rangs moyens pour les ex-aequo (indispensable : la ReLU cree beaucoup de zeros communs ;
    un argsort(argsort) les classerait par position et gonflerait artificiellement la correlation)."""
    from scipy.stats import spearmanr
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def test_adebayo_randomisation_derniere_couche(trained_model, volume):
    """Reinitialiser la tete de sortie d'un modele ENTRAINE doit changer fortement les cartes de tous les niveaux
    (la cible Seg-Grad-CAM et les gradients dependent de la tete)."""
    cls = 1
    c = MultiLevelCAM(trained_model, MOCK_LAYERS)
    try:
        ref, npred, _ = c(volume, [cls])
    finally:
        c.close()
    assert npred[cls] > 0
    rnd = copy.deepcopy(trained_model)
    _randomize(rnd.head)
    c2 = MultiLevelCAM(rnd, MOCK_LAYERS)
    try:
        cur, _, _ = c2(volume, [cls])
    finally:
        c2.close()
    for lv in MOCK_LAYERS:
        a, b = ref[cls][lv], cur[cls][lv]
        assert ssim_3d(a, a) == pytest.approx(1.0)       # controle positif
        assert a.std() > 0, lv                             # la carte d'origine est informative
        assert ssim_3d(a, b) < 0.5, (lv, ssim_3d(a, b))
        if b.std() == 0:
            # tete aleatoire -> classe plus predite -> masque cible vide -> carte uniforme :
            # explication detruite, Spearman indefini (meme cas que le NaN du script Kaggle)
            continue
        assert abs(_spearman(a, b)) < 0.5, (lv, _spearman(a, b))


def test_adebayo_controle_negatif_meme_poids(trained_model, volume):
    """Une copie NON randomisee doit donner des cartes identiques (sinon le test precedent ne prouve rien)."""
    cls = 1
    outs = []
    for m in (trained_model, copy.deepcopy(trained_model)):
        c = MultiLevelCAM(m, MOCK_LAYERS)
        try:
            outs.append(c(volume, [cls])[0][cls])
        finally:
            c.close()
    for lv in MOCK_LAYERS:
        assert ssim_3d(outs[0][lv], outs[1][lv]) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ 6. metriques
def test_metriques_carte_parfaite_et_uniforme():
    g = np.zeros(SHAPE, bool)
    g[8:16, 8:16, 4:8] = True
    parfait = g.astype(np.float32)
    m = all_metrics(parfait, g)
    assert m["in_organ_ratio"] == pytest.approx(1.0)
    assert m["pointing_accuracy"] == 1.0
    assert m["centroid_distance_mm"] == pytest.approx(0.0, abs=1e-9)
    uni = np.ones(SHAPE, np.float32)
    mu = all_metrics(uni, g)
    assert mu["in_organ_ratio"] == pytest.approx(mu["organ_baseline"])   # carte uniforme = baseline
    assert mu["spatial_entropy"] == pytest.approx(1.0)


def test_ssim_proprietes():
    a = np.random.default_rng(0).random(SHAPE)
    assert ssim_3d(a, a) == pytest.approx(1.0)
    assert ssim_3d(a, 1 - a) < 0


# ------------------------------------------------------------------ 7. couches reelles E3 / E2 (MONAI)
@pytest.mark.parametrize("family", ["unet", "segresnet"])
def test_couches_appariees_existent_dans_les_vraies_architectures(family):
    monai_nets = pytest.importorskip("monai.networks.nets")
    if family == "unet":   # E3 : configs/unet_fair.yaml (num_res_units=0, InstanceNorm)
        net = monai_nets.UNet(spatial_dims=3, in_channels=1, out_channels=5, channels=(48, 96, 192, 384, 832),
                              strides=(2, 2, 2, 2), num_res_units=0, norm="instance")
        expected_params = 18_802_973
    else:                  # E2 : configs/default.yaml
        net = monai_nets.SegResNet(spatial_dims=3, in_channels=1, out_channels=5, init_filters=32,
                                   blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
                                   norm=("GROUP", {"num_groups": 8}))
        expected_params = 18_796_101
    assert sum(p.numel() for p in net.parameters()) == expected_params
    net.eval()
    c = MultiLevelCAM(net, NS["LAYERS"][family])
    try:
        maps, _, shapes = c(torch.rand(1, 1, 32, 32, 16, generator=torch.Generator().manual_seed(3)), [1])
    finally:
        c.close()
    res = {k: v[1] for k, v in shapes.items()}   # premiere dimension spatiale
    if family == "unet":
        assert res == {"enc_profond": 2, "dec_demi": 16, "dec_pre_tete": 16, "sortie": 32}
        assert shapes["enc_profond"][0] == 832 and shapes["dec_demi"][0] == 96
    else:
        assert res == {"enc_profond": 4, "dec_demi": 16, "dec_pre_tete": 32, "sortie": 32}
        assert shapes["enc_profond"][0] == 256 and shapes["dec_pre_tete"][0] == 32
    assert all(h.shape == (32, 32, 16) for h in maps[1].values())
