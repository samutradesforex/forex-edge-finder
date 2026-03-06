# Forex Edge Finder — Terminal Redesign Plan

## Vision
A beautiful, world-class backtesting terminal built around **autonomous strategy discovery**.
The system finds and validates edges 24/7. You come back, review what it found, and pick winners.

## Current Problems
1. **Manual backtesting dominates** — Discovery is the last tab, feels like an afterthought
2. **Information overload** — Overview tab alone has 18 metrics + 3 charts. Analytics has 12+ sections
3. **Redundant data** — Account summary appears in both Overview and Account Sim tabs. Win/loss shown 3 times
4. **No strategy extensibility** — Adding a strategy requires editing 3 files with copy-paste patterns
5. **No edge-centric workflow** — Can't click a discovered edge to see its full backtest details
6. **1900-line monolith** — app.py has everything: styling, data, charts, logic

## Architecture Changes

### A. Strategy Registry (`engine/strategies/`)
```
engine/strategies/
├── __init__.py          # Registry + base class
├── sweeps.py            # Liquidity sweeps
├── inducement.py        # Inducement traps
├── stop_hunts.py        # Stop hunts
├── ema_crossover.py     # EMA crossover
├── rsi_reversal.py      # RSI reversal
├── breakout.py          # Swing breakout
├── fvg_entry.py         # FVG fill entry
└── ob_bounce.py         # Order block bounce
```

Each strategy is a class:
```python
class EMAcrossover(Strategy):
    name = "ema_crossover"
    display_name = "EMA Crossover (21/50)"
    category = "trend"                    # For UI grouping
    params = ["rr_ratio", "min_confluence"]  # Which params matter

    def detect(self, df, **kwargs) -> List[InducementSignal]:
        ...  # Move from liquidity.py
```

Adding a new strategy = create 1 file with 1 class. Auto-registered.

### B. App Reorganization — 4 Tabs Instead of 8

#### Tab 1: DASHBOARD (Home)
The landing page. Shows system status at a glance.
- Discovery status bar (running/paused/idle + progress)
- Key stats: total edges found, validated, best edge score
- Top 5 validated edges (mini-table, clickable)
- Data freshness indicators (last cache update per pair)
- Quick actions: Start Discovery, View All Edges

#### Tab 2: DISCOVERY
Full discovery control panel.
- Start/Stop/Configure controls
- Live progress with auto-refresh (st.fragment, already built)
- Full edge table with filters (pair, strategy, interval, validated)
- Edge comparison charts (score distribution, strategy breakdown)
- Export validated edges (CSV/JSON)

#### Tab 3: EDGE EXPLORER
Deep-dive into any discovered edge. **This is the key new feature.**
- Select an edge from dropdown (or click from Dashboard)
- Full backtest replay: equity curve, trade log, metrics
- Walk-forward validation results side-by-side (IS vs OOS)
- Re-run with modified params (tweak and compare)
- Mark as "favorite" / "rejected" for personal curation

#### Tab 4: BACKTEST LAB
Manual testing — for custom exploration and experimentation.
- Pair/interval/strategy selector
- Parameter controls
- Run backtest + see results
- Compact layout: metrics + equity curve + trade log (one scrollable page)
- "Save as Edge" button to add to edge collection

### C. Modular App Structure
```
app.py                  → Main shell (tabs, sidebar, theme)  ~150 lines
ui/
├── __init__.py
├── theme.py            → CSS + color constants + chart layout
├── dashboard.py        → Tab 1: Dashboard
├── discovery.py        → Tab 2: Discovery control
├── edge_explorer.py    → Tab 3: Edge deep-dive
├── backtest_lab.py     → Tab 4: Manual backtesting
└── components.py       → Shared: section(), edge_badge(), fmt_pf(), charts
```

### D. What Gets Cut / Collapsed
| Current | New |
|---------|-----|
| Overview (18 metrics, 3 charts) | Backtest Lab: 6 key metrics + equity curve |
| Analytics (12 sections!) | Edge Explorer: show on-demand per edge |
| Account Sim (duplicate metrics) | Merged into Backtest Lab as expandable |
| Chart (price + overlays) | Edge Explorer: embedded mini-chart |
| Trade Log (full table) | Backtest Lab: integrated below metrics |
| Optimizer (grid search) | Removed — Discovery IS the optimizer |
| Scanner (all pairs) | Removed — Discovery scans all pairs already |
| Auto-Discovery | Promoted to Tab 2: Discovery |

**Key principle:** The Optimizer and Scanner are subsets of what Discovery already does. Remove them as separate features — Discovery replaces both.

### E. Sidebar Simplification
Current sidebar: 30+ controls. New sidebar:
- **Mode toggle**: Discovery / Backtest Lab
- In Discovery mode: min trades, min PF, start/stop
- In Backtest Lab: pair, interval, strategy, key params, RUN button
- Advanced settings in expander (exits, sizing, confluence)

## Implementation Order

### Phase 1: Strategy Registry (backend refactor)
1. Create `engine/strategies/__init__.py` with base class + registry
2. Extract each strategy from `liquidity.py` into its own file
3. Update `backtester.py` dispatch to use registry
4. Update `discovery.py` to use registry
5. Tests pass — no UI changes yet

### Phase 2: UI Modules (split app.py)
1. Extract CSS/theme to `ui/theme.py`
2. Extract shared components to `ui/components.py`
3. Create `ui/dashboard.py`, `ui/discovery.py`, `ui/edge_explorer.py`, `ui/backtest_lab.py`
4. Slim app.py to ~150 lines (tab shell only)

### Phase 3: New UI (the visible change)
1. Build Dashboard tab
2. Build Discovery tab (mostly exists, clean up)
3. Build Edge Explorer (new feature)
4. Build Backtest Lab (consolidated from old tabs)
5. Remove old Scanner + Optimizer tabs

## What Stays the Same
- All engine code (backtester, sizing, structure, loader)
- Data caching system
- Walk-forward validation
- Discovery engine core logic
- Test suite (extend, don't rewrite)
