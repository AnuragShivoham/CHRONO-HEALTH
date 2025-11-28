// booster_min.js
// Fallback, deterministic JS predictor so the function can run while you generate
// a real converted XGBoost predictor. This uses labels.json to set the output length
// and computes deterministic logits from the input features.
//
// Replace this file with the true converted model JS (one-file predictor) when ready.

import labels from "./labels.json" assert { type: "json" };

/**
 * Deterministic pseudo-weight generator:
 * - Produces a weight for (labelIndex, featureIndex) via a simple hash function.
 * - Deterministic and cheap; no external libs.
 */
function weightFor(labelIdx, featIdx) {
  // small deterministic pseudo-random function
  // uses prime multipliers and mod to keep numbers stable.
  const a = 1610612741; // large odd constant
  const b = 805306457;
  // combine indices into a seed-like value
  const seed = (labelIdx + 1) * 1315423911 ^ (featIdx + 1) * 2654435761;
  // apply some mixing and scale down
  const mixed = ((seed * a + b) >>> 0) % 100000;
  // map to small value in [-0.5, 0.5]
  return (mixed / 100000) - 0.5;
}

/**
 * predictXGB(features)
 * - features: array of numbers (already scaled by index.ts)
 * - returns: array of logits (length = labels.length)
 *
 * This is a deterministic placeholder — replace with real converted booster code.
 */
export function predictXGB(features) {
  if (!Array.isArray(features)) {
    throw new Error("predictXGB expects an array of numbers");
  }

  const nLabels = Array.isArray(labels) ? labels.length : 3;
  const logits = new Array(nLabels).fill(0.0);

  // dot-product-like deterministic aggregation
  for (let i = 0; i < nLabels; i++) {
    let s = 0.0;
    // accumulate using deterministic weights
    for (let j = 0; j < features.length; j++) {
      const w = weightFor(i, j);
      const v = typeof features[j] === "number" ? features[j] : 0;
      s += w * v;
    }
    // small bias per label to avoid identical logits
    const bias = ((i * 37) % 7 - 3) * 0.02;
    logits[i] = s + bias;
  }

  return logits;
}
