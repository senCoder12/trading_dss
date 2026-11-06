import { useEffect, useState } from 'react';
import { api } from '../api/client';

// Registry is static for the session — fetch once, share across components.
let cache = null;
let inflight = null;

function load() {
  if (cache) return Promise.resolve(cache);
  if (!inflight) {
    inflight = api.getIndices()
      .then((res) => {
        const map = {};
        for (const i of res?.indices ?? []) map[i.id] = i;
        cache = map;
        return map;
      })
      .catch(() => {
        inflight = null;
        return {};
      });
  }
  return inflight;
}

/** Map of index_id → registry entry ({ display_name, sector_category, has_options, … }). */
export function useIndexMeta() {
  const [meta, setMeta] = useState(cache ?? {});
  useEffect(() => {
    let alive = true;
    load().then((m) => { if (alive) setMeta(m); });
    return () => { alive = false; };
  }, []);
  return meta;
}

/** Human label for an index id, falling back to a prettified id. */
export function indexLabel(meta, id) {
  return meta?.[id]?.display_name ?? id?.replace(/_/g, ' ');
}
