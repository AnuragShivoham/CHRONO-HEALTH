import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatRequest {
  prompt: string;
  history: Message[];
}

interface ChatResponse {
  assistant_response: string;
  suggested_actions: string[];
  insights_from_data: any;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const supabase = createClient(supabaseUrl, serviceKey);

    // --- Verify token sent from client ---
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Missing authorization" }), {
        status: 401,
        headers: corsHeaders,
      });
    }

    const token = authHeader.replace("Bearer ", "").trim();
    const {
      data: { user },
      error: userError,
    } = await supabase.auth.getUser(token);

    if (!user || userError) {
      return new Response(JSON.stringify({ error: "Invalid user" }), {
        status: 401,
        headers: corsHeaders,
      });
    }

    const { prompt = "", history = [] }: ChatRequest = await req.json();

    // --- Fetch health data for insights ---
    const since = new Date();
    since.setDate(since.getDate() - 7);

    const { data: healthData } = await supabase
      .from("health_records")
      .select("*")
      .eq("user_id", user.id)
      .gte("recorded_at", since.toISOString())
      .order("recorded_at", { ascending: false });

    const insights = calculateHealthInsights(healthData || []);

    // --- Build safe conversational AI reply ---
    const response = generateSafeResponse(prompt, history, insights);

    const chatResponse: ChatResponse = {
      assistant_response: response.message,
      suggested_actions: response.suggestions,
      insights_from_data: insights,
    };

    return new Response(JSON.stringify(chatResponse), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("assistant error:", e);
    return new Response(JSON.stringify({ error: "internal error" }), {
      status: 500,
      headers: corsHeaders,
    });
  }
});

/* ---------------- HELPERS ---------------- */

function calculateHealthInsights(rows: any[]) {
  const out = {
    averageSteps: 0,
    averageHeartRate: 0,
    averageSleep: 0,
    averageSpo2: 0,
    lastRecorded: null as string | null,
  };

  if (!rows || rows.length === 0) return out;

  const steps = rows.filter((x) => x.data_type === "steps");
  const hr = rows.filter((x) => x.data_type === "heart_rate");
  const sleep = rows.filter((x) => x.data_type === "sleep_session");
  const spo2 = rows.filter((x) => x.data_type === "spo2");

  out.averageSteps =
    steps.length > 0
      ? steps.reduce((s, r) => s + (r.value ?? 0), 0) / steps.length
      : 0;

  out.averageHeartRate =
    hr.length > 0
      ? hr.reduce((s, r) => s + (r.value ?? 0), 0) / hr.length
      : 0;

  out.averageSleep =
    sleep.length > 0
      ? sleep.reduce((s, r) => s + (r.value ?? 0), 0) / sleep.length
      : 0;

  out.averageSpo2 =
    spo2.length > 0
      ? spo2.reduce((s, r) => s + (r.value ?? 0), 0) / spo2.length
      : 0;

  out.lastRecorded = rows[0]?.recorded_at ?? null;

  return out;
}

function generateSafeResponse(prompt: string, history: Message[], insights: any) {
  const txt = prompt.toLowerCase();
  const suggestions: string[] = [];
  let message = "";

  if (txt.includes("fever") || txt.includes("cough") || txt.includes("pain")) {
    message =
      "I can't diagnose conditions, but I can help interpret your recent health data.";
    suggestions.push("Review health trends", "Consult a healthcare provider");
  } else if (txt.includes("steps")) {
    message = `You're averaging ${Math.round(
      insights.averageSteps
    )} steps daily. Staying active is great!`;
    suggestions.push("Show weekly activity");
  } else if (txt.includes("heart")) {
    message = `Your average heart rate is ${Math.round(
      insights.averageHeartRate
    )} bpm.`;
    suggestions.push("View heart rate history");
  } else {
    message =
      "How can I assist you with your symptoms, data, or wellness insights?";
  }

  message +=
    "\n\nRemember, this is general information only—not a medical diagnosis.";

  return { message, suggestions };
}
