# 🖥️ AegisMind Frontend Monorepo (Turborepo)

A Turborepo-powered monorepo containing the client applications for the **AI Data Readiness & Model Recommendation Platform**:

1. **`dashboard` (`apps/dashboard`)**: Full data diagnosis, 0–100 Data Health Score gauge, explainable AI recommendation checklist, before/after diffs, and AutoML candidate model benchmarking. Runs on `http://localhost:3000`.
2. **`landing` (`apps/landing`)**: Modern landing page showcasing the platform's capabilities, architecture, and workflow with a direct launch CTA to the diagnostic dashboard. Runs on `http://localhost:3001`.

---

## 📁 Monorepo Structure

```
frontend/
├── turbo.json                 # Turborepo pipeline configuration
├── package.json               # Root workspace manifest & Turborepo scripts
├── apps/
│   ├── dashboard/             # Complete pre-ML diagnostic dashboard
│   │   ├── package.json       # @aegismind/dashboard
│   │   ├── next.config.js
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── app/
│   │       ├── components/    # 10 modular UI components
│   │       └── services/      # FastAPI client
│   └── landing/               # AegisMind modern landing page template
│       ├── package.json       # @aegismind/landing
│       ├── next.config.js
│       ├── tsconfig.json
│       └── src/
│           └── app/
```

---

## 🛠️ Quick Commands

```bash
# Navigate to frontend
cd frontend

# Install all workspace dependencies
npm install

# Run both Dashboard (port 3000) and Landing Page (port 3001) concurrently
npm run dev

# Run only the Diagnostic Dashboard
npm run dev:dashboard

# Run only the Landing Page
npm run dev:landing

# Build all applications
npm run build

# Check TypeScript across all workspaces
npm run check-types
```
