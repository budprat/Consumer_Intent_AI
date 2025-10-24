# Human Purchase Intent - Web Application

Modern Next.js web application for testing purchase intent using Semantic Similarity Rating (SSR).

## Overview

This web application provides an intuitive interface for product managers and marketers to:
- Create custom purchase intent surveys
- View real-time survey results
- Compare multiple products side-by-side (A/B testing)
- Access detailed statistical analysis

## Features

✨ **Core Features:**
- **Survey Creation**: Multi-step form with product details, cohort configuration, and demographic targeting
- **Real-Time Polling**: Automatic status updates for running surveys (3-second intervals)
- **SSR Rating Display**: Color-coded purchase intent ratings (1-5 scale) with 90% test-retest reliability
- **Distribution Visualization**: Interactive charts showing probability distributions
- **A/B Testing**: Side-by-side comparison of multiple surveys with detailed metrics
- **Responsive Design**: Mobile-first UI that works on all devices

🎯 **Technical Highlights:**
- Server-Side Rendering (SSR) with Next.js 15
- Type-safe API client with full TypeScript coverage
- React Query for intelligent caching and polling
- shadcn/ui components for accessible, customizable UI
- Real-time updates without page refresh

## Technology Stack

- **Framework**: [Next.js 15.5.6](https://nextjs.org/) (App Router)
- **Language**: [TypeScript 5](https://www.typescriptlang.org/)
- **UI Library**: [React 19.1.0](https://react.dev/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Components**: [shadcn/ui](https://ui.shadcn.com/)
- **State Management**: [TanStack Query v5.90.5](https://tanstack.com/query/latest) (React Query)
- **Form Handling**: [React Hook Form 7.65.0](https://react-hook-form.com/)
- **Validation**: [Zod 3.25.76](https://zod.dev/)
- **Charts**: [Recharts 2.15.4](https://recharts.org/)
- **Icons**: [Lucide React 0.546.0](https://lucide.dev/)
- **Date Formatting**: [date-fns 3.6.0](https://date-fns.org/)
- **Theming**: [next-themes 0.4.6](https://github.com/pacocoursey/next-themes)

## Prerequisites

- **Node.js**: 18.17 or later
- **npm**: 9 or later (or yarn/pnpm)
- **FastAPI Backend**: Running on `http://localhost:8000`

## Installation

1. **Navigate to web-app directory:**
   ```bash
   cd Human_Purchase_Intent/web-app
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   ```

   Edit `.env.local` and configure:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

## Running the Application

### Development Mode

Start the development server with hot-reload:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

**Note:** This project uses Turbopack for faster development builds.

### Production Build

Build and run the production version:

```bash
npm run build
npm start
```

### Linting

Run ESLint to check code quality:

```bash
npm run lint
```

## Environment Variables

Create a `.env.local` file with the following variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEXT_PUBLIC_API_URL` | FastAPI backend URL | `http://localhost:8000` | Yes |

**Note:** Variables prefixed with `NEXT_PUBLIC_` are exposed to the browser.

## Project Structure

```
web-app/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with navigation
│   ├── page.tsx           # Dashboard (home page)
│   ├── providers.tsx      # React Query provider
│   ├── globals.css        # Global styles
│   ├── surveys/
│   │   ├── new/
│   │   │   └── page.tsx   # Create new survey
│   │   └── [id]/
│   │       └── page.tsx   # Survey results (dynamic route)
│   └── compare/
│       └── page.tsx       # Compare surveys (A/B testing)
├── components/
│   ├── surveys/           # Survey-specific components
│   │   ├── survey-form.tsx
│   │   ├── survey-card.tsx
│   │   ├── ssr-rating-badge.tsx
│   │   ├── survey-status.tsx
│   │   └── distribution-chart.tsx
│   ├── shared/            # Reusable components
│   │   ├── page-header.tsx
│   │   ├── loading-spinner.tsx
│   │   └── error-display.tsx
│   └── ui/                # shadcn/ui primitives
├── hooks/                 # Custom React hooks
│   ├── use-surveys.ts
│   ├── use-survey.ts
│   ├── use-survey-results.ts
│   ├── use-create-survey.ts
│   └── use-survey-status.ts
├── lib/
│   ├── api.ts             # Type-safe API client
│   ├── types.ts           # TypeScript type definitions
│   ├── query-client.ts    # React Query configuration
│   └── utils.ts           # Utility functions
└── public/                # Static assets
```

## API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8000`.

### Key Endpoints Used:

- `GET /api/v1/surveys` - List all surveys
- `POST /api/v1/surveys` - Create new survey
- `GET /api/v1/surveys/{id}` - Get survey details
- `GET /api/v1/surveys/{id}/results` - Get survey results
- `GET /api/v1/tasks/{task_id}` - Poll task status

### CORS Configuration

Ensure the FastAPI backend allows requests from `http://localhost:3000`:

```python
# In FastAPI main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development Workflow

### Creating New Components

We use shadcn/ui for components. To add a new component:

```bash
npx shadcn@latest add [component-name]
```

### Type Safety

All API responses and component props are fully typed. Types are defined in:
- `lib/types.ts` - API types matching FastAPI Pydantic models
- Component files - Component-specific prop types

### State Management Pattern

- **Server State**: React Query hooks in `hooks/`
- **Form State**: React Hook Form with Zod validation
- **UI State**: Local component state (useState)

### Styling Conventions

- Tailwind CSS 4 utility classes
- shadcn/ui design system
- Responsive breakpoints: `sm:` (640px), `md:` (768px), `lg:` (1024px)

## Deployment

### Prerequisites for Production

1. Update `NEXT_PUBLIC_API_URL` to production API URL
2. Configure CORS on backend for production domain
3. Set up proper SSL certificates
4. Configure environment variables in hosting platform

### Deployment Platforms

This Next.js app can be deployed to:
- **Vercel** (recommended) - Zero config deployment
- **Netlify** - Static export or SSR
- **Docker** - Containerized deployment
- **Self-hosted** - Node.js server

### Build Output

```bash
npm run build
```

Creates optimized production build in `.next/` directory.

## Troubleshooting

### Common Issues

**1. CORS Errors**
- Ensure FastAPI backend has CORS middleware configured
- Check `NEXT_PUBLIC_API_URL` matches backend URL
- Verify backend is accessible from your browser

**2. API Connection Failed**
- Verify FastAPI backend is running on port 8000
- Check network connectivity
- Inspect browser console for error details
- Ensure `.env.local` is properly configured

**3. Build Errors**
- Clear `.next` cache: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules package-lock.json && npm install`
- Verify Node.js version: `node --version` (should be 18.17+)

**4. Turbopack Issues**
- If experiencing issues with Turbopack, remove `--turbopack` flag from scripts temporarily
- Report issues to Next.js team as Turbopack is still in beta

## Contributing

1. Create a feature branch from `main`
2. Make your changes with proper TypeScript types
3. Ensure linting passes: `npm run lint`
4. Build succeeds: `npm run build`
5. Submit a pull request

## License

This project is part of the Human Purchase Intent SSR system.

## Support

For issues or questions:
- Check existing issues in GitHub repository
- Review FastAPI backend logs for API errors
- Inspect browser console for client-side errors
- Verify environment variables are correctly configured

## Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Query Documentation](https://tanstack.com/query/latest/docs/react/overview)
- [shadcn/ui Components](https://ui.shadcn.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
