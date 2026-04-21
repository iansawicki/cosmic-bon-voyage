import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import OpenAI from "https://deno.land/x/openai@v4.24.0/mod.ts"

serve(async (req) => {
  try {
    const { prompt, artists } = await req.json()

    const openai = new OpenAI({
      apiKey: Deno.env.get("OPENAI_API_KEY")
    })

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `
You are a music curator.

Generate a short, evocative station name (2–4 words max).

Style:
- poetic
- immersive
- minimal
- not generic
- no emojis
- no quotes

Avoid:
- "playlist"
- "mix"
- "station"

Focus on mood, atmosphere, and feeling.
`
        },
        {
          role: "user",
          content: `
User prompt: ${prompt}
Artists: ${artists}
`
        }
      ],
      temperature: 0.8,
      max_tokens: 20
    })

    const title = completion.choices[0].message.content?.trim()

    return new Response(JSON.stringify({ title }), {
      headers: { "Content-Type": "application/json" }
    })

  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500
    })
  }
})
