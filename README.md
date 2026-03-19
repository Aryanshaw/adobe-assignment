# Agent Backend

This backend uses FastAPI + SQLAlchemy + Alembic with Postgres.

## Prerequisites

1. Sync Python dependencies:

```bash
uv sync
```

2. Ensure `.env` has a DB URL:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname?sslmode=require
```

## Migration Commands

Run from `apps/agent`:

- Apply all migrations:

```bash
npm run db:upgrade
```

- Roll back one migration:

```bash
npm run db:downgrade
```

- Create a new migration (autogenerate):

```bash
npm run db:revision -- "describe change"
```

After creating a revision, apply it:

```bash
npm run db:upgrade
```

## Typical Workflow

1. Update models in `models/`.
2. Generate revision:

```bash
npm run db:revision -- "describe change"
```

3. Review generated file in `alembic/versions/`.
4. Apply migration:

```bash
npm run db:upgrade
```

## Notes

- Alembic reads `DATABASE_URL` from `.env`.
- `sslmode` in the DB URL is normalized for asyncpg in `config/db.py`.
- App startup does not auto-create tables. Migrations are the source of truth.
