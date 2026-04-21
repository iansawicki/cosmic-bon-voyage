import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";

const app = new Hono();

app.use("*", logger(console.log));

app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

function getSupabaseClient() {
  return createClient(
    Deno.env.get("SUPABASE_URL") ?? "",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    },
  );
}

// Branded verification email: admin.generateLink (magiclink) + Resend template.
// Secrets: RESEND_API_KEY, optional RESEND_FROM, APP_AUTH_REDIRECT_URI.
// Template: RESEND_VERIFICATION_TEMPLATE_ID (default Cosmic verify template),
//   RESEND_VERIFICATION_LINK_VARIABLE must match the variable name in the Resend editor (default VERIFY_LINK).
// Client must send the user's JWT.
app.post("/*", async (c) => {
  try {
    const accessToken = c.req.header("Authorization")?.split(" ")[1];
    if (!accessToken) {
      return c.json({ error: "Unauthorized", code: "unauthorized" }, 401);
    }

    const supabase = getSupabaseClient();
    const { data: userData, error: userErr } = await supabase.auth.getUser(
      accessToken,
    );
    if (userErr || !userData?.user?.email) {
      return c.json({ error: "Unauthorized", code: "unauthorized" }, 401);
    }

    const user = userData.user;
    const meta = user.user_metadata as Record<string, unknown> | undefined;
    const appFlag = meta?.app_email_verified;
    if (appFlag === true || appFlag === "true") {
      return c.json({ ok: true, alreadyVerified: true });
    }

    const resendKey = Deno.env.get("RESEND_API_KEY");
    if (!resendKey || resendKey.trim() === "") {
      console.error("RESEND_API_KEY is not set");
      return c.json(
        { error: "Email delivery is not configured", code: "missing_resend" },
        503,
      );
    }

    const redirectTo =
      Deno.env.get("APP_AUTH_REDIRECT_URI")?.trim() ||
      "co.cosmicradio.cosmicradio://login-callback";

    const { data: linkData, error: linkErr } = await supabase.auth.admin
      .generateLink({
        type: "magiclink",
        email: user.email,
        options: { redirectTo },
      });

    if (linkErr) {
      console.error("generateLink:", linkErr);
      return c.json(
        { error: linkErr.message ?? "Could not create verification link" },
        400,
      );
    }

    const props = linkData?.properties as
      | { action_link?: string; href?: string }
      | undefined;
    const actionLink = props?.action_link ?? props?.href;
    if (!actionLink || typeof actionLink !== "string") {
      console.error("generateLink: missing action_link", linkData);
      return c.json(
        { error: "Could not create verification link" },
        500,
      );
    }

    const from = Deno.env.get("RESEND_FROM")?.trim() ||
      "Cosmic Radio <noreply@mail.cosmicradio.co>";

    const defaultVerificationTemplateId =
      "c5baed0d-1a7e-4d55-ab71-bdf744a5b1b5";
    const templateId =
      Deno.env.get("RESEND_VERIFICATION_TEMPLATE_ID")?.trim() ||
      defaultVerificationTemplateId;
    const linkVariable =
      Deno.env.get("RESEND_VERIFICATION_LINK_VARIABLE")?.trim() ||
      "VERIFY_LINK";

    const emailPayload: Record<string, unknown> = {
      from,
      to: [user.email],
      subject: "Verify your email for Cosmic Radio",
      template: {
        id: templateId,
        variables: {
          [linkVariable]: actionLink,
        },
      },
    };

    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${resendKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(emailPayload),
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error("Resend HTTP", res.status, errText);
      return c.json(
        { error: "Could not send verification email" },
        502,
      );
    }

    return c.json({ ok: true });
  } catch (error) {
    console.error("send-verification-email:", error);
    return c.json(
      { error: "Internal server error", code: "internal_error" },
      500,
    );
  }
});

Deno.serve(app.fetch);
