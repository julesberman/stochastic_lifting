# OmniChat — Code Style Guide

For use by AI coding assistants (Claude Code, etc.) when writing or modifying code in this project.

---

## Philosophy

Readable first, production-grade second. This is a v1 built for one person to understand, run, and iterate on fast. Optimize for clarity and iteration speed, not team scalability or enterprise patterns.

---

## General Rules

- **Functions over classes.** Use plain functions, plain objects, and simple modules. Reach for a class only when you genuinely need stateful instances with behavior — not as an organizational default.
- **Direct over clever.** If a straightforward 10-line implementation works, prefer it over a 4-line version that requires mental gymnastics to read. No magic, no metaprogramming tricks.
- **Minimal abstraction.** Don't generalize until there are 3+ concrete use cases. No factory functions, no DI containers, no strategy patterns. Build the specific thing.
- **Flat over nested.** Early returns, guard clauses, and flat control flow. Avoid deep nesting or callback pyramids.
- **Small files.** One concept per file. If a file grows past ~200 lines, split it.

---

## TypeScript (Frontend — Svelte)

```ts
// Use `type` over `interface` unless extending is needed
type Pane = {
  id: string
  provider: Provider
  bounds: Rect
}

// Union types for fixed sets — no enums
type Provider = 'claude' | 'chatgpt' | 'gemini'

// Functions, not classes
function createPane(provider: Provider): Pane { ... }

// Destructure args when there are 3+
function resizePane({ id, width, height }: { id: string; width: number; height: number }) { ... }
```

- Prefer `const` and arrow functions for small utilities, `function` declarations for anything exported or substantial.
- Use Svelte 5 runes (`$state`, `$derived`, `$effect`) idiomatically. Keep component logic in the component; extract to a module only when shared.
- No barrel files (`index.ts` re-exports). Import directly from the source file.

---

## Rust (Backend — Tauri)

```rust
// Keep Tauri commands thin — do the work in helper functions
#[tauri::command]
fn save_layout(state: State<AppState>, layout: Layout) -> Result<(), String> {
    persist_layout(&state.store, &layout).map_err(|e| e.to_string())
}
```

- Use `thiserror` for error types only if there are 3+ distinct error variants. Otherwise `.map_err(|e| e.to_string())` is fine.
- Minimize `unsafe`. For this project there should be zero.
- Keep the Rust layer as thin as possible — it's glue, not business logic.

---

## Comments & Documentation

```ts
// Good: explains *why*, not *what*
// Debounce resize to avoid hammering the webview — it re-layouts on every pixel change
const debouncedResize = debounce(applyResize, 150)

// Bad: narrates the obvious
// Set width to 500
width = 500
```

- Inline comments: short, lowercase, explain intent or non-obvious decisions.
- Doc comments (`/** */` / `///`): one line for public functions. Skip if the name + types already say everything.
- No comment blocks, no ASCII art dividers, no `@param` / `@returns` tags.
- TODOs are fine: `// TODO: handle case where provider changes input selector`.

---

## Error Handling

Keep it light. This is a personal v1, not a distributed system.

- **Do:** A few `assert` or `console.assert` calls to clarify assumptions in critical spots.
- **Do:** Simple sanity checks near core logic if they help you understand what's expected.
- **Don't:** Validate inputs on every function boundary.
- **Don't:** Nest `try/catch` blocks. One `try/catch` at the call site is usually enough.
- **Don't:** Write custom exception classes or detailed error messages nobody will read.

```ts
// Good: assert clarifies a precondition
assert(panes.length <= 4, 'max 4 panes')

// Good: single try/catch at the boundary
async function loadLayout() {
  try {
    const raw = await store.get('layout')
    return raw ? parseLayout(raw) : DEFAULT_LAYOUT
  } catch {
    return DEFAULT_LAYOUT // corrupt store, just reset
  }
}

// Bad: defensive everything
function loadLayout() {
  if (!store) throw new StoreNotInitializedError(...)
  if (typeof store.get !== 'function') throw new InvalidStoreError(...)
  try { ... } catch (e) {
    if (e instanceof JsonParseError) { ... }
    else if (e instanceof FileNotFoundError) { ... }
    else { ... }
  }
}
```

---

## Naming

- Files: `kebab-case.ts`, `kebab-case.svelte`
- Types: `PascalCase`
- Functions / variables: `camelCase`
- Constants: `UPPER_SNAKE` only for true config constants (URLs, limits). Regular `const` bindings stay `camelCase`.
- Be specific: `paneId` not `id`, `providerUrl` not `url` — unless scope is tiny (loop body, arrow function).

---

## Formatting

- Prettier with defaults. Don't bikeshed config — use what `pnpm create tauri-app` scaffolds.
- Single quotes, no semicolons (or whatever Prettier defaults to — just be consistent).
- Max line length: ~100 chars soft limit. Don't contort code to hit it.

---

## Dependencies

- Add a dependency when it saves real complexity (e.g., `svelte-splitpanes`, `debounce`).
- Don't add a dependency for something achievable in <30 lines of code.
- Pin versions in `package.json`. No `^` or `~`.
- Before adding a new dep, check if Tauri or Svelte already provide it.

---

## Things to Avoid

- Barrel files and re-export layers.
- Abstract base classes, inheritance hierarchies.
- `any` in TypeScript (use `unknown` + narrowing if truly needed).
- Over-logging. A few `console.log` calls during dev are fine; don't build a logging framework.
- Config-driven anything. Hardcode provider URLs, pane limits, shortcuts. Extract to config only when there's a real reason.
- Feature flags, A/B test scaffolding, analytics hooks.
- Premature performance optimization. Profile first, optimize second.
