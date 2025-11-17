# Security Guidelines

## Protecting API Keys and Secrets

### ⚠️ NEVER Commit Secrets to Git

**Critical files to keep private:**
- `.env` - Contains all API keys and credentials
- Any files with actual API keys or tokens
- Database credentials
- Production configuration files

### ✅ How to Manage Secrets Safely

1. **Use `.env` file for local development**
   ```bash
   # Copy the template
   cp .env.example .env

   # Edit with your actual credentials
   # This file is in .gitignore and will NOT be committed
   ```

2. **Never hardcode credentials in code**
   ```python
   # ❌ BAD - Never do this!
   api_key = "sk-abc123..."

   # ✅ GOOD - Use environment variables
   from study_query_llm.config import config
   api_key = config.get_provider_config("openai").api_key
   ```

3. **Use environment variables in production**
   - Set environment variables directly in your deployment platform
   - Use secret management services (Azure Key Vault, AWS Secrets Manager, etc.)

4. **Verify `.gitignore` is working**
   ```bash
   # Check what would be committed
   git status

   # .env should NOT appear in the list
   # If it does, it's already in .gitignore so don't git add it
   ```

## If You Accidentally Commit Secrets

If you accidentally commit API keys or secrets:

1. **Rotate the credentials immediately**
   - Go to your provider's dashboard
   - Revoke/delete the exposed key
   - Generate a new key

2. **Remove from Git history** (if needed)
   ```bash
   # Use git-filter-repo or BFG Repo-Cleaner
   # This rewrites history - coordinate with team first
   ```

3. **Update `.env` with new credentials**

## Environment Variables Reference

### Required for Azure OpenAI
```bash
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=<your-endpoint-url>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
```

### Required for OpenAI
```bash
OPENAI_API_KEY=<your-api-key>
```

### Required for Hyperbolic
```bash
HYPERBOLIC_API_KEY=<your-api-key>
```

### Optional
```bash
DATABASE_URL=<connection-string>
LANGFUSE_PUBLIC_KEY=<pk>
LANGFUSE_SECRET_KEY=<sk>
```

## Best Practices

1. **Least Privilege**: Only give credentials the minimum permissions needed
2. **Rotation**: Rotate API keys periodically
3. **Monitoring**: Monitor API usage for unexpected patterns
4. **Separate Environments**: Use different credentials for dev/staging/production
5. **Team Access**: Use a secret management service for team collaboration

## Reporting Security Issues

If you discover a security vulnerability, please email: [your-email@example.com]

**Do not** open public GitHub issues for security vulnerabilities.
