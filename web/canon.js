// Value canonicalization post-processor (JS port of training/canon.py).
//
// Many "errors" in raw model output are value FORMAT mismatches, not
// semantic errors: "3 PM" vs "15:00", "8am" vs "08:00", 24.444 vs 24.4,
// "Saturdays" vs "Saturday". Canonicalizing predicted argument values
// toward the dataset's conventions recovers those. The bench measured
// +2.7 pp oracle lift from this step (PLAN.md Iter 29).

const TIME_12H = /\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b/gi;
const TIME_24H = /\b(\d{1,2}):(\d{2})\b/g;

const DAY_NAMES = new Set([
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
]);

function to24h(h, m, ampm) {
  ampm = ampm.toLowerCase();
  if (ampm === 'a') {
    if (h === 12) h = 0;
  } else if (h !== 12) {
    h += 12;
  }
  return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
}

export function canonicalizeTimeString(s) {
  let out = s.replace(TIME_12H, (_m, hh, mm, ap) =>
    to24h(parseInt(hh, 10), mm ? parseInt(mm, 10) : 0, ap),
  );
  out = out.replace(TIME_24H, (_m, hh, mm) =>
    String(parseInt(hh, 10)).padStart(2, '0') + ':' +
    String(parseInt(mm, 10)).padStart(2, '0'),
  );
  return out;
}

function looksLikeTimeKey(key) {
  const k = key.toLowerCase();
  return ['time', 'alarm', '_at', 'when', 'schedule'].some((t) => k.includes(t));
}

function looksLikeDayKey(key) {
  return ['day', 'days', 'weekday', 'dayofweek', 'day_of_week'].includes(
    key.toLowerCase(),
  );
}

function canonDayToken(tok) {
  const low = tok.toLowerCase();
  if (low.endsWith('s') && DAY_NAMES.has(low.slice(0, -1))) {
    return tok.slice(0, -1);
  }
  return tok;
}

function canonicalizeValue(key, v) {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') {
    const r = Math.round(v * 10) / 10;
    return r;
  }
  if (typeof v === 'string') {
    const t = v.trim();
    if (/^-?\d+$/.test(t)) return parseInt(t, 10);
    if (/^-?\d+\.\d+$/.test(t)) return Math.round(parseFloat(t) * 10) / 10;
    if (looksLikeTimeKey(key) && (TIME_12H.test(t) || TIME_24H.test(t))) {
      TIME_12H.lastIndex = 0;
      TIME_24H.lastIndex = 0;
      return canonicalizeTimeString(t);
    }
    TIME_12H.lastIndex = 0;
    TIME_24H.lastIndex = 0;
    if (looksLikeDayKey(key)) return canonDayToken(t);
    return t;
  }
  if (Array.isArray(v)) return v.map((x) => canonicalizeValue(key, x));
  if (v && typeof v === 'object') return canonicalizeArgs(v);
  return v;
}

export function canonicalizeArgs(args) {
  if (!args || typeof args !== 'object' || Array.isArray(args)) return {};
  const out = {};
  for (const [k, v] of Object.entries(args)) {
    out[k] = canonicalizeValue(k, v);
  }
  return out;
}

// --- Enum value snapping (JS port of canon.py snap_enums, Iter 38) -------
// Snap a predicted argument value to its tool_registry enum member:
// "gym" -> "basement gym", "living_room" -> "living room". Verified +3 pp
// on the synthesis pipeline (PLAN.md Iter 38). Three conservative levels —
// case-insensitive exact, underscore/space-insensitive, unique substring.

function loose(s) {
  return s.trim().toLowerCase().replace(/[\s_]+/g, ' ');
}

export function snapEnumValue(v, enumList) {
  if (typeof v !== 'string' || !Array.isArray(enumList) || enumList.length === 0) {
    return v;
  }
  const lv = loose(v);
  for (const e of enumList) {
    if (typeof e === 'string' && loose(e) === lv) return e;
  }
  const subs = enumList.filter(
    (e) => typeof e === 'string' && (loose(e).includes(lv) || lv.includes(loose(e))),
  );
  return subs.length === 1 ? subs[0] : v;
}

export function snapEnums(name, args, registry) {
  if (!args || typeof args !== 'object' || Array.isArray(args)) return {};
  let enums = {};
  if (typeof name === 'string' && registry && typeof registry === 'object') {
    enums = (registry[name] || {}).enums || {};
  }
  const out = {};
  for (const [k, v] of Object.entries(args)) {
    out[k] = k in enums ? snapEnumValue(v, enums[k]) : v;
  }
  return out;
}

// Full prediction post-process: enum-snap (if a registry is given) then
// value canonicalization.
export function canonicalizeCall(name, args, registry) {
  return canonicalizeArgs(registry ? snapEnums(name, args, registry) : args);
}
