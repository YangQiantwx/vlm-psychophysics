import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";

const root = process.cwd();
const denyDirs = new Set([".git", "dist", "node_modules", "qa_screenshots", ".wrangler"]);
const textExts = new Set([".css", ".html", ".js", ".jsx", ".json", ".md", ".txt"]);
const secretPatterns = [
  /gho_[A-Za-z0-9_]+/g,
  /sk-[A-Za-z0-9_-]{20,}/g,
  /xox[baprs]-[A-Za-z0-9-]+/g,
  /AIza[0-9A-Za-z_-]{20,}/g,
  /BEGIN [A-Z ]*PRIVATE KEY/g,
  /\b(client_secret|refresh_token|access_token|api[_-]?key|password)\b\s*[:=]\s*["'][^"']+["']/gi,
];

function walk(dir, files = []) {
  for (const name of readdirSync(dir)) {
    if (denyDirs.has(name)) continue;
    const path = join(dir, name);
    const stat = statSync(path);
    if (stat.isDirectory()) {
      walk(path, files);
      continue;
    }
    files.push(path);
  }
  return files;
}

const failures = [];

if (!existsSync(join(root, "public", "_headers"))) {
  failures.push("Missing public/_headers.");
}

if (!existsSync(join(root, "public", "robots.txt"))) {
  failures.push("Missing public/robots.txt.");
}

for (const file of walk(root)) {
  if (!textExts.has(extname(file))) continue;
  const text = readFileSync(file, "utf8");
  for (const pattern of secretPatterns) {
    pattern.lastIndex = 0;
    if (pattern.test(text)) {
      failures.push(`Possible secret pattern in ${relative(root, file)}: ${pattern}`);
    }
  }
}

if (existsSync(join(root, "dist"))) {
  const sourcemaps = walk(join(root, "dist")).filter((path) => path.endsWith(".map"));
  if (sourcemaps.length > 0) {
    failures.push(`Public sourcemaps found in dist: ${sourcemaps.map((p) => relative(root, p)).join(", ")}`);
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("Release check passed: headers present, no obvious secrets, no public sourcemaps.");
