// Prerequis : npm install docx@9.5.1 (dans un dossier de travail), puis : node paper/md2docx.js paper/Article_Ablation_6OAR_v1.md . paper/Article_Ablation_6OAR_v1.docx
// Convertit paper/Article_Ablation_6OAR_v1.md en .docx (A4), figures integrees a la fin.
// node md2docx.js <article.md> <racine_projet> <sortie.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, ImageRun, LevelFormat, BorderStyle, Footer, PageNumber,
} = require("docx");

const [, , mdPath, root, outPath] = process.argv;
const lines = fs.readFileSync(mdPath, "utf8").replace(/\r/g, "").split("\n");
const TEXT_W = 11906 - 2 * 1304; // A4, marges 2,3 cm
const FONT = "Calibri";

function runs(text, base = {}) {
  // **gras**, *italique*, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 18, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function table(rows) {
  const cells = rows.map((r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const head = cells[0];
  const body = cells.slice(2);
  const n = head.length;
  const first = n > 4 ? Math.round(TEXT_W * 0.2) : Math.round(TEXT_W / n);
  const rest = Math.floor((TEXT_W - first) / (n - 1));
  const widths = [first, ...Array(n - 1).fill(rest)];
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const size = n > 5 ? 15 : 17;
  const mk = (row, isHead) => new TableRow({
    tableHeader: isHead,
    children: row.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "E8E7E3" } : undefined,
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      children: [new Paragraph({ children: runs(c, { size, bold: isHead || undefined }) })],
    })),
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(head, true), ...body.map((r) => mk(r, false))],
  });
}

const children = [];
let i = 0;
let inFigures = false;
while (i < lines.length) {
  const l = lines[i];
  if (/^\|/.test(l)) {
    const block = [];
    while (i < lines.length && /^\|/.test(lines[i])) block.push(lines[i++]);
    children.push(table(block));
    children.push(new Paragraph({ children: [], spacing: { after: 120 } }));
    continue;
  }
  if (l.startsWith("# ")) children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(l.slice(2)) }));
  else if (l.startsWith("## ")) {
    inFigures = /^## Figures/.test(l);
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(l.slice(3)) }));
  } else if (l.startsWith("### ")) children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(l.slice(4)) }));
  else if (/^---\s*$/.test(l)) children.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBEB8", space: 1 } }, children: [] }));
  else if (inFigures && /^- Fig\./.test(l)) {
    const m = l.match(/^- (Fig\. \d+) — (.*?): `([^`]+)`/);
    const png = path.join(root, m[3] + ".png");
    if (fs.existsSync(png)) {
      const s = pngSize(png);
      const wPx = 620, hPx = Math.round((wPx * s.h) / s.w);
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true,
        children: [new ImageRun({ type: "png", data: s.data, transformation: { width: wPx, height: hPx } })] }));
    }
    children.push(new Paragraph({ spacing: { after: 240 }, children: runs(`**${m[1]}.** ${m[2]}.`, { size: 18 }) }));
  } else if (/^\s*- /.test(l)) {
    const lvl = l.match(/^\s*/)[0].length >= 2 ? 1 : 0;
    children.push(new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(l.replace(/^\s*- /, "")) }));
  } else if (/^\s*\d+\. /.test(l)) {
    const lvl = l.match(/^\s*/)[0].length >= 2 ? 1 : 0;
    children.push(new Paragraph({ numbering: { reference: "num", level: lvl }, children: runs(l.replace(/^\s*\d+\. /, "")) }));
  } else if (l.trim() === "") { /* saut */ }
  else children.push(new Paragraph({ spacing: { after: 120 }, children: runs(l.trim()) }));
  i++;
}

const hstyle = (size, before, after) => ({ run: { font: FONT, size, bold: true, color: "1F1F1E" },
  paragraph: { spacing: { before, after }, keepNext: true } });
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } },
      title: hstyle(32, 0, 200), heading1: hstyle(26, 280, 120), heading2: hstyle(22, 200, 100) },
  },
  numbering: { config: [
    { reference: "bul", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 260 } } } }] },
    { reference: "num", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } },
      { level: 1, format: LevelFormat.LOWER_LETTER, text: "%2.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 300 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1304, bottom: 1304, left: 1304, right: 1304 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("ecrit", outPath, children.length, "blocs"); });
