---
name: security
description: Security review for vibe-coded or AI-generated web apps. Use when the user invokes /security, asks to check, audit, or secure an app, API, or codebase, or asks whether code is safe to deploy. Covers injection, authentication and access control, secrets and tokens, and config hygiene.
---

# Security Review Skill

Use this skill to review an app or code for the common security mistakes found in quickly built, AI-assisted projects. Work through the checklist below, report what you find, and suggest concrete fixes. Do not make changes to the code unless the user asks.

## Workflow

1. **Identify the stack.** Find the language, framework, database, auth method, and where the app is hosted. Note whether it uses a backend, a frontend only, or a backend-as-a-service such as Supabase or Firebase.
2. **Check each category below** against the actual code, config, and environment files. Search for the patterns listed (for example, string-built SQL, `innerHTML`, hardcoded keys, `cors({ origin: "*" })`). Do not report a risk you have not verified in the code.
3. **Report findings** grouped by severity: Critical (exploitable now, such as leaked secrets or SQL injection), High (likely exploitable with some effort, such as missing ownership checks), Medium (hardening gaps, such as no rate limiting), Low (hygiene, such as outdated dependencies).
4. **For each finding**, give the file and line if possible, the risk in one sentence, and the fix.
5. **End with a short summary** of what passed and what needs attention first. If something cannot be verified from the code alone, say so and suggest how the user can check it.

## Checklist

### 1. Injection and Input

- **SQL Injection:** user input concatenated into queries. Fix with parameterized queries or an ORM.
- **Cross-Site Scripting (XSS):** untrusted content rendered as HTML (`innerHTML`, `dangerouslySetInnerHTML`, `v-html`). Fix by escaping output and setting a Content Security Policy.
- **CSRF Protection:** state-changing endpoints without CSRF tokens or with cookies lacking `SameSite`. Fix with tokens and `SameSite=Lax` or `Strict`.
- **File Uploads:** no check of file content, size, or type, or files stored in an executable or public path. Fix with content validation, size limits, random filenames, and safe storage.
- **SSRF:** server-side fetches of user-supplied URLs with no allowlist. Fix by allowlisting domains and blocking private and link-local IP ranges, including cloud metadata addresses.

### 2. Authentication and Access

- **Broken Object Level Authorization (BOLA / IDOR):** endpoints that load a record by ID without checking the requester owns it. Fix with an ownership check on every request.
- **Rate Limiting:** no limits on login, signup, password reset, OTP, or public APIs. Fix with per-IP or per-user limits.
- **Password Storage:** passwords stored in plain text, or hashed with MD5, SHA-1, or unsalted SHA-256. Fix with bcrypt, scrypt, or Argon2.
- **Multi-Factor Authentication:** no second factor, especially for admin accounts. Recommend MFA for all accounts and required MFA for admins.
- **Server-Side Permissions:** access control that exists only in frontend code or hidden UI. Fix by enforcing checks on the server.
- **Row Level Security (RLS):** in Supabase or PostgreSQL, tables holding user data without RLS enabled, or with overly broad policies. Fix by enabling RLS and writing per-operation policies.

### 3. Secrets and Tokens

- **JWT Secrets:** hardcoded, weak, or short signing keys; `alg: none` accepted; tokens without expiry. Fix with a long random secret from an environment variable, strict algorithm checks, and short expiry.
- **API Secrets:** API keys in frontend code, `.env` files committed to Git, or keys prefixed for public exposure (such as `NEXT_PUBLIC_`, `VITE_`) that hold private values. Fix by keeping secrets on the server and rotating any that were exposed.
- **Local Storage:** tokens or sensitive data in `localStorage` or `sessionStorage`. Fix with httpOnly, Secure cookies.
- **Default Credentials:** default admin accounts, passwords, or database credentials still present. Fix by changing them and removing unused accounts.

### 4. Config and Hygiene

- **CORS:** wildcard origins, or reflecting the request `Origin` header with credentials enabled. Fix by allowing only known origins.
- **Webhook Signatures:** webhook handlers that act without verifying the signature. Fix by verifying the HMAC or provider signature first.
- **Exposed Source Maps:** `.map` files deployed to production. Fix by removing them or restricting access.
- **Sensitive Data Exposure:** API responses returning password hashes, tokens, or internal fields; stack traces shown to users; secrets or PII in logs. Fix by filtering response fields, hiding stack traces, and scrubbing logs.
- **Vulnerable Dependencies:** outdated or known-vulnerable packages. Check lockfiles and run `npm audit`, `pip-audit`, or the equivalent for the stack.

## Output format

Use short headings for each severity level. Under each finding, write the file location, the risk, and the fix as brief prose or a short list. Keep the summary to a few sentences. Avoid long preambles.

## Boundaries

- Review and advise. Do not edit files or rotate secrets unless the user explicitly asks.
- If the user shares a secret while asking for a review, advise them to rotate it, and do not repeat it back.
- This checklist covers common issues, not a full penetration test. Say so when the user is relying on the review for a production launch with sensitive data.
