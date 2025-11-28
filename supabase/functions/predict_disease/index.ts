import { serve } from "https://deno.land/std@0.182.0/http/server.ts";

// -------------------------
// LOAD FILES AT RUNTIME
// (NOT USING IMPORTS)
// -------------------------

async function loadModelFiles() {
  const scaler = JSON.parse(await Deno.readTextFile("./scaler.json"));
  const labels = JSON.parse(await Deno.readTextFile("./labels.json"));

  // booster_min.js exports a function → we eval it manually
  const boosterCode = await Deno.readTextFile("./booster_min.js");
  const predictXGB = new Function(`${boosterCode}; return predictXGB;`)();

  return { scaler, labels, predictXGB };
}

// ---- SCALER ----
function scaleInput(arr, scaler) {
  return arr.map((v, i) => (v - scaler.mean[i]) / (scaler.std[i] || 1));
}

// ---- SOFTMAX ----
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

serve(async (req) => {
  try {
    if (req.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    const body = await req.json();
    const symptoms = body.symptoms;

    if (!Array.isArray(symptoms)) {
      return new Response(JSON.stringify({ error: "symptoms must be array" }), {
        status: 400,
      });
    }

    // Load everything
    const { scaler, labels, predictXGB } = await loadModelFiles();

    const scaled = scaleInput(symptoms, scaler);

    // Get logits
    const logits = predictXGB(scaled);

    const probs = softmax(logits);
    const bestIndex = probs.indexOf(Math.max(...probs));
    const prediction = labels[bestIndex] ?? bestIndex;

    return new Response(
      JSON.stringify({
        prediction,
        probabilities: probs,
      }),
      { headers: { "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("predict_disease error:", err);
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
    });
  }
});
