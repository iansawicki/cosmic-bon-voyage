import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { resolveAuthEmailPreflight } from "../_shared/auth_admin_lookup.ts";

const app = new Hono();

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
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

// Helper to create Supabase client with service role (Edge / server: no persisted session)
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

// Health check endpoint
app.get("/make-server-3c2fd8d1/health", (c) => {
  return c.json({ status: "ok" });
});

// ============================================================================
// AUTHENTICATION ENDPOINTS
// ============================================================================

// Sign up endpoint
app.post("/make-server-3c2fd8d1/auth/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();

    if (!email || !password) {
      return c.json({ error: "Email and password are required" }, 400);
    }

    const supabase = getSupabaseClient();

    const { data, error } = await supabase.auth.admin.createUser({
      email,
      password,
      user_metadata: { name: name || '' },
      email_confirm: true
    });

    if (error) {
      console.error('Sign up error:', error);
      
      if (error.message.includes('already been registered') || error.message.includes('email_exists')) {
        return c.json({ 
          error: 'An account with this email already exists. Please sign in instead.' 
        }, 409);
      }
      
      return c.json({ error: error.message }, 400);
    }

    // Initialize user preferences in database
    if (data.user) {
      await supabase.from('user_preferences').insert({
        user_id: data.user.id,
        favorites: [],
        settings: {
          skipLimit: 6,
          createdAt: new Date().toISOString()
        },
        onboarding_complete: false
      });
    }

    return c.json({ 
      success: true, 
      user: {
        id: data.user?.id,
        email: data.user?.email,
        name: data.user?.user_metadata?.name
      }
    });
  } catch (error) {
    console.error('Sign up error:', error);
    return c.json({ error: 'Internal server error during sign up' }, 500);
  }
});

// Email preflight endpoint for unified auth entry.
app.post("/make-server-b5ae0368/auth/email-preflight", async (c) => {
  try {
    const body = await c.req.json();
    const rawEmail = String(body?.email ?? '').trim().toLowerCase();

    if (!rawEmail) {
      return c.json({ error: 'Email is required', code: 'missing_email' }, 400);
    }
    const looksLikeEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(rawEmail);
    if (!looksLikeEmail) {
      return c.json({ error: 'Invalid email address', code: 'invalid_email' }, 400);
    }

    const supabase = getSupabaseClient();
    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
    const preflight = await resolveAuthEmailPreflight(
      supabase,
      supabaseUrl,
      serviceKey,
      rawEmail,
    );

    if (!preflight.ok) {
      return c.json(
        { error: 'Failed to check email status', code: 'preflight_failed' },
        500,
      );
    }

    if (!preflight.exists) {
      return c.json({
        ok: true,
        email: rawEmail,
        exists: false,
        emailVerified: false,
        status: 'new_user',
      });
    }

    return c.json({
      ok: true,
      email: rawEmail,
      exists: true,
      emailVerified: preflight.emailVerified,
      status: preflight.emailVerified
        ? 'existing_verified'
        : 'existing_unverified',
    });
  } catch (error) {
    console.error('Email preflight error:', error);
    return c.json(
      { error: 'Internal server error while checking email', code: 'internal_error' },
      500,
    );
  }
});

// Get user data endpoint
app.get("/make-server-3c2fd8d1/auth/user", async (c) => {
  try {
    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    
    if (!accessToken) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    const supabase = getSupabaseClient();
    const { data: { user }, error } = await supabase.auth.getUser(accessToken);

    if (error || !user) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    // Get user preferences from database
    const { data: prefs } = await supabase
      .from('user_preferences')
      .select('favorites')
      .eq('user_id', user.id)
      .single();

    return c.json({ 
      user: {
        id: user.id,
        email: user.email,
        name: user.user_metadata?.name,
        favorites: prefs?.favorites || []
      }
    });
  } catch (error) {
    console.error('Get user error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Save user favorites endpoint
app.post("/make-server-3c2fd8d1/user/favorites", async (c) => {
  try {
    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    
    if (!accessToken) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    const supabase = getSupabaseClient();
    const { data: { user }, error } = await supabase.auth.getUser(accessToken);

    if (error || !user) {
      console.error('Authorization error while saving favorites:', error);
      return c.json({ error: 'Unauthorized' }, 401);
    }

    const { favorites } = await c.req.json();

    // Save favorites to user_preferences table
    const { error: updateError } = await supabase
      .from('user_preferences')
      .upsert({
        user_id: user.id,
        favorites: favorites,
        updated_at: new Date().toISOString()
      });

    if (updateError) {
      console.error('Save favorites error:', updateError);
      return c.json({ error: 'Failed to save favorites' }, 500);
    }

    return c.json({ success: true });
  } catch (error) {
    console.error('Save favorites error:', error);
    return c.json({ error: 'Internal server error while saving favorites' }, 500);
  }
});

// ============================================================================
// ONBOARDING ENDPOINTS
// ============================================================================

// Submit onboarding preferences (shared handler)
const handleOnboardingSubmit = async (c: any) => {
  try {
    const { preferences, userId, sessionId } = await c.req.json();

    if (!preferences || !Array.isArray(preferences)) {
      return c.json({ error: 'Preferences array is required' }, 400);
    }

    const supabase = getSupabaseClient();
    const id = `onboarding:${Date.now()}:${Math.random().toString(36).substr(2, 9)}`;

    // Insert into onboarding_submissions table
    const { error: insertError } = await supabase
      .from('onboarding_submissions')
      .insert({
        id,
        user_id: userId || null,
        session_id: sessionId || null,
        preferences: preferences
      });

    if (insertError) {
      console.error('Onboarding submission error:', insertError);
      return c.json({ error: 'Failed to save onboarding data' }, 500);
    }

    // Update user_preferences if user is authenticated
    if (userId) {
      const { error: updateError } = await supabase
        .from('user_preferences')
        .upsert({
          user_id: userId,
          onboarding_complete: true,
          updated_at: new Date().toISOString()
        });

      if (updateError) {
        console.error('User preferences update error:', updateError);
      }
    }

    return c.json({ 
      success: true, 
      id,
      preferences,
      createdAt: new Date().toISOString()
    });
  } catch (error) {
    console.error('Submit onboarding error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
};

// Submit onboarding preferences (legacy endpoint)
app.post("/make-server-3c2fd8d1/onboarding/submit", handleOnboardingSubmit);

// Submit onboarding preferences (frontend expects this endpoint)
app.post("/make-server-3c2fd8d1/onboarding/preferences", handleOnboardingSubmit);

// Check if user completed onboarding (legacy endpoint)
app.get("/make-server-3c2fd8d1/onboarding/check/:userId", async (c) => {
  try {
    const userId = c.req.param('userId');
    const supabase = getSupabaseClient();

    const { data, error } = await supabase
      .from('user_preferences')
      .select('onboarding_complete')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') {
      console.error('Check onboarding error:', error);
      return c.json({ completed: false });
    }

    return c.json({ completed: data?.onboarding_complete || false });
  } catch (error) {
    console.error('Check onboarding error:', error);
    return c.json({ completed: false });
  }
});

// Check onboarding status (frontend expects this endpoint)
app.get("/make-server-3c2fd8d1/onboarding/status/:userId", async (c) => {
  try {
    const userId = c.req.param('userId');
    const supabase = getSupabaseClient();

    const { data, error } = await supabase
      .from('user_preferences')
      .select('onboarding_complete')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') {
      console.error('Check onboarding status error:', error);
      return c.json({ completed: false });
    }

    return c.json({ completed: data?.onboarding_complete || false });
  } catch (error) {
    console.error('Check onboarding status error:', error);
    return c.json({ completed: false });
  }
});

// Mark onboarding as complete (frontend expects this endpoint)
app.post("/make-server-3c2fd8d1/onboarding/complete", async (c) => {
  try {
    const { userId } = await c.req.json();

    if (!userId) {
      return c.json({ error: 'userId is required' }, 400);
    }

    const supabase = getSupabaseClient();

    const { error } = await supabase
      .from('user_preferences')
      .upsert({
        user_id: userId,
        onboarding_complete: true,
        updated_at: new Date().toISOString()
      });

    if (error) {
      console.error('Mark onboarding complete error:', error);
      return c.json({ error: 'Failed to mark onboarding complete' }, 500);
    }

    console.log('Onboarding marked complete for user:', userId);

    return c.json({ 
      success: true,
      message: 'Onboarding completion saved',
      userId 
    });
  } catch (error) {
    console.error('Mark onboarding complete error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// ============================================================================
// FEEDBACK ENDPOINTS
// ============================================================================

// Submit feedback
app.post("/make-server-3c2fd8d1/feedback/submit", async (c) => {
  try {
    const { email, message, userId } = await c.req.json();

    if (!email || !message) {
      return c.json({ error: 'Email and message are required' }, 400);
    }

    const supabase = getSupabaseClient();
    const id = `feedback:${Date.now()}:${Math.random().toString(36).substr(2, 9)}`;

    const { error } = await supabase
      .from('feedback_submissions')
      .insert({
        id,
        user_id: userId || null,
        email,
        message,
        status: 'new'
      });

    if (error) {
      console.error('Feedback submission error:', error);
      return c.json({ error: 'Failed to submit feedback' }, 500);
    }

    return c.json({ success: true, id });
  } catch (error) {
    console.error('Submit feedback error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get all feedback
app.get("/make-server-3c2fd8d1/feedback/all", async (c) => {
  try {
    const supabase = getSupabaseClient();

    const { data, error } = await supabase
      .from('feedback_submissions')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Get feedback error:', error);
      return c.json({ error: 'Failed to retrieve feedback' }, 500);
    }

    return c.json({ feedback: data || [] });
  } catch (error) {
    console.error('Get feedback error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// ============================================================================
// ANALYTICS ENDPOINTS
// ============================================================================

// Track analytics event
app.post("/make-server-3c2fd8d1/analytics/track", async (c) => {
  try {
    const { eventName, properties, userId } = await c.req.json();

    if (!eventName) {
      return c.json({ error: 'eventName is required' }, 400);
    }

    const supabase = getSupabaseClient();
    const eventId = `event:${Date.now()}:${Math.random().toString(36).substr(2, 9)}`;
    const timestamp = new Date();
    const date = timestamp.toISOString().split('T')[0];
    const hour = timestamp.getHours();
    const dayOfWeek = timestamp.getDay();

    // Insert event into events table with retry logic
    try {
      const { error: eventError } = await supabase
        .from('events')
        .insert({
          id: eventId,
          event_name: eventName,
          user_id: userId || null,
          session_id: properties?.sessionId || null,
          properties: properties || {},
          timestamp: timestamp.toISOString(),
          date,
          hour,
          day_of_week: dayOfWeek
        });

      if (eventError) {
        console.error('Event tracking error:', eventError);
        // Don't fail the request - just log and continue
        // This allows the app to continue working even if analytics fails
      }
    } catch (insertError) {
      console.error('Event insert failed (connection issue):', insertError);
      // Continue execution - analytics failure shouldn't break the app
    }

    // Track daily active user (non-critical, swallow errors)
    if (userId) {
      try {
        await supabase
          .from('daily_active_users')
          .upsert({
            date,
            user_id: userId,
            event_count: 1,
            first_seen_at: timestamp.toISOString(),
            last_seen_at: timestamp.toISOString()
          }, {
            onConflict: 'date,user_id',
            ignoreDuplicates: false
          });
      } catch (dauError) {
        console.error('Daily active users update failed:', dauError);
        // Non-critical, continue
      }

      // Update user last activity (non-critical, swallow errors)
      try {
        await supabase
          .from('user_preferences')
          .upsert({
            user_id: userId,
            last_activity: timestamp.toISOString(),
            updated_at: timestamp.toISOString()
          }, {
            onConflict: 'user_id',
            ignoreDuplicates: false
          });
      } catch (prefsError) {
        console.error('User preferences update failed:', prefsError);
        // Non-critical, continue
      }
    }

    // Update session data (non-critical, swallow errors)
    if (properties?.sessionId) {
      try {
        const { data: existingSession } = await supabase
          .from('sessions')
          .select('*')
          .eq('id', properties.sessionId)
          .single();

        if (existingSession) {
          // Update existing session
          const updates: any = {
            last_activity_time: timestamp.toISOString(),
            event_count: (existingSession.event_count || 0) + 1,
            updated_at: timestamp.toISOString()
          };

          if (eventName === 'session_end' && properties.duration) {
            updates.end_time = timestamp.toISOString();
            updates.duration_ms = properties.duration;
          }

          await supabase
            .from('sessions')
            .update(updates)
            .eq('id', properties.sessionId);
        } else {
          // Create new session
          await supabase
            .from('sessions')
            .insert({
              id: properties.sessionId,
              user_id: userId || null,
              start_time: timestamp.toISOString(),
              last_activity_time: timestamp.toISOString(),
              event_count: 1,
              metadata: {
                userAgent: properties.userAgent,
                screenResolution: properties.screenResolution,
                timezone: properties.timezone
              }
            });
        }
      } catch (sessionError) {
        console.error('Session update failed:', sessionError);
        // Non-critical, continue
      }
    }

    console.log('Analytics event tracked:', eventName, 'for user:', userId || 'anonymous');

    // Always return success - analytics failures shouldn't break the app
    return c.json({ success: true, eventId });
  } catch (error) {
    console.error('Track analytics error:', error);
    // Return success even on error to prevent app disruption
    // The error is logged for debugging
    return c.json({ 
      success: true, 
      eventId: `fallback:${Date.now()}`,
      warning: 'Event tracked with degraded functionality'
    });
  }
});

// Get analytics summary
app.get("/make-server-3c2fd8d1/analytics/summary", async (c) => {
  try {
    const days = parseInt(c.req.query('days') || '7');
    const supabase = getSupabaseClient();
    
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const startDateStr = startDate.toISOString().split('T')[0];

    // Get daily active users
    const { data: dauData } = await supabase
      .from('daily_active_users')
      .select('date, user_id')
      .gte('date', startDateStr)
      .order('date', { ascending: false });

    const dailyActiveUsers = dauData?.reduce((acc: any[], row) => {
      const existing = acc.find(item => item.date === row.date);
      if (existing) {
        existing.count++;
      } else {
        acc.push({ date: row.date, count: 1 });
      }
      return acc;
    }, []) || [];

    // Get event counts by type
    const { data: eventCountsData } = await supabase
      .from('events')
      .select('event_name')
      .gte('date', startDateStr);

    const eventCounts: Record<string, number> = {};
    eventCountsData?.forEach(event => {
      eventCounts[event.event_name] = (eventCounts[event.event_name] || 0) + 1;
    });

    // Get session statistics
    const { data: sessionsData } = await supabase
      .from('sessions')
      .select('duration_ms, end_time')
      .gte('start_time', startDate.toISOString());

    const totalSessions = sessionsData?.length || 0;
    const completedSessions = sessionsData?.filter(s => s.end_time).length || 0;
    const avgDuration = sessionsData && sessionsData.length > 0
      ? sessionsData.reduce((sum, s) => sum + (s.duration_ms || 0), 0) / sessionsData.length
      : 0;

    // Get total unique users
    const uniqueUserIds = new Set(dauData?.map(d => d.user_id) || []);
    const totalUniqueUsers = uniqueUserIds.size;

    return c.json({
      period: `${days} days`,
      dailyActiveUsers,
      totalSessions,
      completedSessions,
      averageSessionDuration: avgDuration,
      eventCounts,
      summary: {
        totalEvents: eventCountsData?.length || 0,
        totalDAU: dailyActiveUsers.reduce((sum: number, day: any) => sum + day.count, 0),
        avgDAU: dailyActiveUsers.length > 0 
          ? Math.round(dailyActiveUsers.reduce((sum: number, day: any) => sum + day.count, 0) / dailyActiveUsers.length)
          : 0,
        totalUniqueUsers
      }
    });
  } catch (error) {
    console.error('Get analytics summary error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get engagement metrics
app.get("/make-server-3c2fd8d1/analytics/engagement", async (c) => {
  try {
    const supabase = getSupabaseClient();

    // Get track likes
    const { data: trackLikes } = await supabase
      .from('events')
      .select('properties')
      .eq('event_name', 'track_liked')
      .order('timestamp', { ascending: false });

    // Aggregate most liked tracks
    const trackLikeCounts: Record<string, any> = {};
    trackLikes?.forEach(event => {
      const trackId = event.properties?.trackId;
      if (trackId) {
        if (!trackLikeCounts[trackId]) {
          trackLikeCounts[trackId] = {
            trackId,
            trackName: event.properties?.trackName,
            artistName: event.properties?.artistName,
            count: 0
          };
        }
        trackLikeCounts[trackId].count++;
      }
    });

    const mostLikedTracks = Object.values(trackLikeCounts)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 10);

    // Get station likes
    const { data: stationLikes } = await supabase
      .from('events')
      .select('properties')
      .eq('event_name', 'station_liked')
      .order('timestamp', { ascending: false });

    const stationLikeCounts: Record<string, any> = {};
    stationLikes?.forEach(event => {
      const stationId = event.properties?.stationId;
      if (stationId) {
        if (!stationLikeCounts[stationId]) {
          stationLikeCounts[stationId] = {
            stationId,
            stationName: event.properties?.stationName,
            count: 0
          };
        }
        stationLikeCounts[stationId].count++
;
      }
    });

    const mostLikedStations = Object.values(stationLikeCounts)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 10);

    // Get skip data
    const { data: trackSkips } = await supabase
      .from('events')
      .select('properties')
      .eq('event_name', 'track_skipped');

    const trackSkipCounts: Record<string, any> = {};
    trackSkips?.forEach(event => {
      const trackId = event.properties?.trackId;
      if (trackId) {
        if (!trackSkipCounts[trackId]) {
          trackSkipCounts[trackId] = {
            trackId,
            trackName: event.properties?.trackName,
            artistName: event.properties?.artistName,
            count: 0
          };
        }
        trackSkipCounts[trackId].count++;
      }
    });

    const mostSkippedTracks = Object.values(trackSkipCounts)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 10);

    // Get hourly and daily distribution
    const { data: allEvents } = await supabase
      .from('events')
      .select('hour, day_of_week');

    const hourlyDistribution: Record<number, number> = {};
    const dailyDistribution: Record<number, number> = {};

    allEvents?.forEach(event => {
      hourlyDistribution[event.hour] = (hourlyDistribution[event.hour] || 0) + 1;
      dailyDistribution[event.day_of_week] = (dailyDistribution[event.day_of_week] || 0) + 1;
    });

    const peakHours = Object.entries(hourlyDistribution)
      .map(([hour, count]) => ({ hour: parseInt(hour), count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const peakDays = Object.entries(dailyDistribution)
      .map(([day, count]) => ({ 
        day: parseInt(day), 
        dayName: dayNames[parseInt(day)], 
        count 
      }))
      .sort((a, b) => b.count - a.count);

    return c.json({
      mostLikedTracks,
      mostLikedStations,
      mostSkippedTracks,
      peakHours,
      peakDays,
      hourlyDistribution,
      dailyDistribution
    });
  } catch (error) {
    console.error('Get engagement metrics error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get retention metrics
app.get("/make-server-3c2fd8d1/analytics/retention", async (c) => {
  try {
    const days = parseInt(c.req.query('days') || '30');
    const supabase = getSupabaseClient();

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    // Get all daily active users
    const { data: dauData } = await supabase
      .from('daily_active_users')
      .select('*')
      .gte('date', startDate.toISOString().split('T')[0])
      .order('date', { ascending: false });

    // Calculate retention (simplified)
    const dailyActiveUsers = dauData?.reduce((acc: any[], row) => {
      const existing = acc.find(item => item.date === row.date);
      if (existing) {
        existing.count++;
        existing.users.push(row.user_id);
      } else {
        acc.push({ date: row.date, count: 1, users: [row.user_id] });
      }
      return acc;
    }, []) || [];

    // Get session stats
    const { data: sessionsData } = await supabase
      .from('sessions')
      .select('user_id')
      .gte('start_time', startDate.toISOString());

    const uniqueUsers = new Set(sessionsData?.map(s => s.user_id).filter(Boolean));

    return c.json({
      dailyActiveUsers,
      sessions: {
        totalSessions: sessionsData?.length || 0,
        uniqueUsers: uniqueUsers.size,
        averageSessionsPerUser: uniqueUsers.size > 0 
          ? (sessionsData?.length || 0) / uniqueUsers.size 
          : 0
      }
    });
  } catch (error) {
    console.error('Get retention metrics error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get performance metrics
app.get("/make-server-3c2fd8d1/analytics/performance", async (c) => {
  try {
    const supabase = getSupabaseClient();

    // Get errors
    const { data: errors } = await supabase
      .from('events')
      .select('*')
      .eq('event_name', 'error')
      .order('timestamp', { ascending: false })
      .limit(50);

    const errorsByType: Record<string, number> = {};
    errors?.forEach(error => {
      const errorType = error.properties?.errorType || 'unknown';
      errorsByType[errorType] = (errorsByType[errorType] || 0) + 1;
    });

    // Get playback issues
    const { data: playbackIssues } = await supabase
      .from('events')
      .select('*')
      .eq('event_name', 'playback_issue')
      .order('timestamp', { ascending: false });

    const issuesByType: Record<string, number> = {};
    playbackIssues?.forEach(issue => {
      const issueType = issue.properties?.issueType || 'unknown';
      issuesByType[issueType] = (issuesByType[issueType] || 0) + 1;
    });

    // Get buffering data
    const { data: buffering } = await supabase
      .from('events')
      .select('properties')
      .eq('event_name', 'buffering');

    const totalBufferingTime = buffering?.reduce((sum, event) => 
      sum + (event.properties?.bufferingDuration || 0), 0
    ) || 0;

    const avgBufferingTime = buffering && buffering.length > 0
      ? totalBufferingTime / buffering.length
      : 0;

    // Calculate error rate
    const { count: totalEvents } = await supabase
      .from('events')
      .select('*', { count: 'exact', head: true });

    const errorRate = totalEvents && totalEvents > 0
      ? ((errors?.length || 0) / totalEvents) * 100
      : 0;

    return c.json({
      errors: {
        total: errors?.length || 0,
        byType: errorsByType,
        errorRate: errorRate.toFixed(2),
        recent: errors?.slice(0, 10) || []
      },
      playbackIssues: {
        total: playbackIssues?.length || 0,
        byType: issuesByType
      },
      buffering: {
        total: buffering?.length || 0,
        totalTime: totalBufferingTime,
        averageTime: avgBufferingTime
      }
    });
  } catch (error) {
    console.error('Get performance metrics error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get search analytics
app.get("/make-server-3c2fd8d1/analytics/searches", async (c) => {
  try {
    const days = parseInt(c.req.query('days') || '7');
    const supabase = getSupabaseClient();

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const { data: searches } = await supabase
      .from('events')
      .select('*')
      .eq('event_name', 'search_query')
      .gte('timestamp', startDate.toISOString())
      .order('timestamp', { ascending: false });

    // Aggregate search queries
    const queryCounts: Record<string, any> = {};
    let totalQueryLength = 0;

    searches?.forEach(search => {
      const query = search.properties?.query?.toLowerCase();
      if (query) {
        totalQueryLength += query.length;
        if (!queryCounts[query]) {
          queryCounts[query] = {
            query,
            count: 0,
            firstSeen: search.timestamp,
            lastSeen: search.timestamp
          };
        }
        queryCounts[query].count++;
        queryCounts[query].lastSeen = search.timestamp;
      }
    });

    const topSearches = Object.values(queryCounts)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 20);

    const recentSearches = searches?.slice(0, 50).map(s => ({
      query: s.properties?.query,
      timestamp: s.timestamp,
      userId: s.user_id,
      source: s.properties?.source
    })) || [];

    // Searches by hour
    const searchesByHour: Record<number, number> = {};
    searches?.forEach(search => {
      searchesByHour[search.hour] = (searchesByHour[search.hour] || 0) + 1;
    });

    const peakSearchHours = Object.entries(searchesByHour)
      .map(([hour, count]) => ({ hour: parseInt(hour), count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    return c.json({
      totalSearches: searches?.length || 0,
      uniqueSearches: Object.keys(queryCounts).length,
      averageQueryLength: searches && searches.length > 0 
        ? totalQueryLength / searches.length 
        : 0,
      topSearches,
      recentSearches,
      searchesByHour,
      peakSearchHours
    });
  } catch (error) {
    console.error('Get search analytics error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get user sessions
app.get("/make-server-3c2fd8d1/analytics/sessions", async (c) => {
  try {
    const userId = c.req.query('userId');
    const limit = parseInt(c.req.query('limit') || '50');
    const supabase = getSupabaseClient();

    let query = supabase
      .from('sessions')
      .select('*')
      .order('start_time', { ascending: false })
      .limit(limit);

    if (userId) {
      query = query.eq('user_id', userId);
    }

    const { data: sessions } = await query;

    // Get events for each session
    const sessionsWithEvents = await Promise.all(
      (sessions || []).map(async (session) => {
        const { data: events } = await supabase
          .from('events')
          .select('event_name, timestamp, properties')
          .eq('session_id', session.id)
          .order('timestamp', { ascending: true });

        return {
          ...session,
          events: events || []
        };
      })
    );

    return c.json({ sessions: sessionsWithEvents });
  } catch (error) {
    console.error('Get sessions error:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// ============================================================================
// Start Server
// ============================================================================

Deno.serve(app.fetch);