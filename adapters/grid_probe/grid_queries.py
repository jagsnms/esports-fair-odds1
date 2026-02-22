# GRID PROBE V1 — GraphQL query strings for GRID Open Access.
# Aligned with introspected schema: allSeries (Central), seriesState (Series State).

# ---------------------------------------------------------------------------
# A) Central Data — allSeries connection (required: orderBy, orderDirection)
# ---------------------------------------------------------------------------
# GRID PROBE V1 — filter (SeriesFilter) supports: live, startTimeScheduled, updatedAt, titleIds, etc.
# Series node fields (id, title, startedAt) may need one more introspection if errors return.
QUERY_FIND_CS2_SERIES = """
query FindCS2Series($orderBy: SeriesOrderBy!, $orderDirection: OrderDirection!, $first: Int, $filter: SeriesFilter) {
  allSeries(
    orderBy: $orderBy,
    orderDirection: $orderDirection,
    first: $first,
    filter: $filter
  ) {
    totalCount
    edges {
      cursor
      node {
        id
        title {
          name
          nameShortened
        }
        type
        updatedAt
        startTimeScheduled
        teams {
          baseInfo {
            id
            name
            nameShortened
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
  }
}
"""

# GRID PROBE V1 — minimal fallback if Series node field names (e.g. title, startedAt) are wrong
QUERY_FIND_CS2_SERIES_MINIMAL = """
query FindCS2SeriesMinimal($orderBy: SeriesOrderBy!, $orderDirection: OrderDirection!, $first: Int, $filter: SeriesFilter) {
  allSeries(
    orderBy: $orderBy,
    orderDirection: $orderDirection,
    first: $first,
    filter: $filter
  ) {
    totalCount
    edges {
      cursor
      node {
        id
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
  }
}
"""

# ---------------------------------------------------------------------------
# B) Live Series State — seriesState(id: ID!)
# ---------------------------------------------------------------------------
# GRID PROBE V1
# Nested game/team/player fields may require schema alignment; keep conservative.
# GRID PROBE V1 — title is type Title! (object); subselection from introspected fields (nameShortened).
QUERY_SERIES_STATE = """
query SeriesState($id: ID!) {
  seriesState(id: $id) {
    version
    id
    title {
      nameShortened
    }
    format
    started
    finished
    forfeited
    valid
    updatedAt
    startedAt
    duration
    teams {
      id
    }
    games {
      id
    }
  }
}
"""

# GRID PROBE V2 — Series State query with ONLY V2 subset fields (docs/GRID_CS2_NORMALIZED_FEATURE_CONTRACT.md).
# __typename + inline fragments on CSGO/CS2 concrete types (API may return Cs2 for title cs2); minimal path; no polling/retries.
QUERY_SERIES_STATE_RICH = """
query SeriesStateRich($id: ID!) {
  seriesState(id: $id) {
    __typename
    id
    valid
    updatedAt
    started
    finished
    title {
      nameShortened
    }
    teams {
      __typename
      ... on SeriesTeamStateCsgo {
        id
        score
      }
      ... on SeriesTeamStateCs2 {
        id
        score
      }
    }
    games {
      __typename
      sequenceNumber
      started
      finished
      map {
        name
      }
      clock {
        currentSeconds
        ticking
        type
        ticksBackwards
      }
      segments {
        sequenceNumber
        teams {
          __typename
          ... on SegmentTeamStateCsgo {
            id
            won
          }
          ... on SegmentTeamStateCs2 {
            id
            won
          }
        }
      }
      teams {
        __typename
        ... on GameTeamStateCsgo {
          id
          score
          side
          money
          loadoutValue
          players {
            __typename
            ... on GamePlayerStateCsgo {
              id
              alive
              currentHealth
              currentArmor
              money
              loadoutValue
            }
          }
        }
        ... on GameTeamStateCs2 {
          id
          score
          side
          money
          loadoutValue
          players {
            __typename
            ... on GamePlayerStateCsgo {
              id
              alive
              currentHealth
              currentArmor
              money
              loadoutValue
            }
            ... on GamePlayerStateCs2 {
              id
              alive
              currentHealth
              currentArmor
              money
              loadoutValue
            }
          }
        }
      }
    }
  }
}
"""
