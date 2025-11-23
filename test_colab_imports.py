"""
Test script to verify Colab setup imports work correctly.

This simulates what would happen in Colab to ensure all imports are valid.
"""

import sys
import os

# Simulate Colab environment
print("Testing Colab setup imports...")
print("=" * 60)

# Test 1: Check dependencies can be imported
print("\n1. Testing dependency imports...")
try:
    import panel
    print("   [OK] panel")
except ImportError as e:
    print(f"   [FAIL] panel: {e}")

try:
    import dotenv
    print("   [OK] python-dotenv")
except ImportError as e:
    print(f"   [FAIL] python-dotenv: {e}")

try:
    import openai
    print("   [OK] openai")
except ImportError as e:
    print(f"   [FAIL] openai: {e}")

try:
    import tenacity
    print("   [OK] tenacity")
except ImportError as e:
    print(f"   [FAIL] tenacity: {e}")

try:
    import sqlalchemy
    print("   [OK] sqlalchemy")
except ImportError as e:
    print(f"   [FAIL] sqlalchemy: {e}")

try:
    import pandas
    print("   [OK] pandas")
except ImportError as e:
    print(f"   [FAIL] pandas: {e}")

# Test 2: Check our package imports
print("\n2. Testing package imports...")
try:
    from study_query_llm.config import config
    print("   [OK] study_query_llm.config")
except ImportError as e:
    print(f"   [FAIL] study_query_llm.config: {e}")

try:
    from study_query_llm.db.connection import DatabaseConnection
    print("   [OK] study_query_llm.db.connection")
except ImportError as e:
    print(f"   [FAIL] study_query_llm.db.connection: {e}")

try:
    from study_query_llm.providers.factory import ProviderFactory
    print("   [OK] study_query_llm.providers.factory")
except ImportError as e:
    print(f"   [FAIL] study_query_llm.providers.factory: {e}")

try:
    from panel_app.app import create_app, serve_app
    print("   [OK] panel_app.app")
except ImportError as e:
    print(f"   [FAIL] panel_app.app: {e}")

# Test 3: Check environment variable setup
print("\n3. Testing environment variable setup...")
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"

try:
    from study_query_llm.config import config
    azure_config = config.get_provider_config("azure")
    print("   [OK] Environment variables loaded")
    print(f"     Endpoint: {azure_config.endpoint}")
    print(f"     Deployment: {azure_config.deployment_name}")
except Exception as e:
    print(f"   [FAIL] Environment setup failed: {e}")

# Test 4: Check database initialization
print("\n4. Testing database initialization...")
try:
    from study_query_llm.db.connection import DatabaseConnection
    from study_query_llm.config import config
    
    db = DatabaseConnection("sqlite:///:memory:")  # Use in-memory for test
    db.init_db()
    print("   [OK] Database initialization works")
except Exception as e:
    print(f"   [FAIL] Database initialization failed: {e}")

# Test 5: Check app creation
print("\n5. Testing app creation...")
try:
    from panel_app.app import create_app
    app = create_app()
    print("   [OK] App creation works")
    print(f"     App type: {type(app)}")
except Exception as e:
    print(f"   [FAIL] App creation failed: {e}")

print("\n" + "=" * 60)
print("Import tests complete!")
print("\nNote: This tests imports only. Full Colab testing requires:")
print("  - Running in actual Colab environment")
print("  - Testing Panel server startup")
print("  - Testing URL generation")
print("  - Testing app functionality in browser")

