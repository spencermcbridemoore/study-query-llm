"""
End-to-End Test Verification Script

This script verifies that inferences run through the GUI are properly stored
in the database and can be queried.
"""

import sys
from datetime import datetime, timedelta
from study_query_llm.config import config
from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.inference_repository import InferenceRepository


def check_database_initialized():
    """Verify database exists and has the correct schema."""
    print("="*60)
    print("Checking Database Initialization")
    print("="*60)
    
    try:
        db = DatabaseConnection(config.database.connection_string)
        
        # Try to query the table
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            count = repo.get_total_count()
            print(f"[OK] Database connection successful")
            print(f"[OK] Table 'inference_runs' exists")
            print(f"[OK] Current inference count: {count}")
            return True
    except Exception as e:
        print(f"[FAIL] Database check failed: {e}")
        print("\nAttempting to initialize database...")
        try:
            db = DatabaseConnection(config.database.connection_string)
            db.init_db()
            print("[OK] Database initialized successfully")
            return True
        except Exception as e2:
            print(f"[FAIL] Failed to initialize: {e2}")
            return False


def check_recent_inferences(minutes=5):
    """Check for inferences created in the last N minutes."""
    print("\n" + "="*60)
    print(f"Checking for Recent Inferences (last {minutes} minutes)")
    print("="*60)
    
    try:
        db = DatabaseConnection(config.database.connection_string)
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            
            # Get all recent inferences
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=minutes)
            
            recent = repo.query_inferences(
                date_range=(start_date, end_date),
                limit=100
            )
            
            if not recent:
                print(f"[WARN] No inferences found in the last {minutes} minutes")
                print("   Run an inference through the GUI to test.")
                return False
            
            print(f"[OK] Found {len(recent)} inference(s) in the last {minutes} minutes:\n")
            
            for inf in recent[:5]:  # Show first 5
                print(f"  ID: {inf.id}")
                print(f"  Provider: {inf.provider}")
                print(f"  Prompt: {inf.prompt[:50]}..." if len(inf.prompt) > 50 else f"  Prompt: {inf.prompt}")
                print(f"  Response: {inf.response[:50]}..." if len(inf.response) > 50 else f"  Response: {inf.response}")
                print(f"  Tokens: {inf.tokens}")
                print(f"  Latency: {inf.latency_ms:.2f} ms" if inf.latency_ms else "  Latency: N/A")
                print(f"  Created: {inf.created_at}")
                print()
            
            return True
            
    except Exception as e:
        print(f"[FAIL] Error checking recent inferences: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_analytics_data():
    """Verify analytics data is available."""
    print("="*60)
    print("Checking Analytics Data")
    print("="*60)
    
    try:
        db = DatabaseConnection(config.database.connection_string)
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            
            # Get provider stats
            stats = repo.get_provider_stats()
            
            if not stats:
                print("[WARN] No provider statistics available")
                print("   Run some inferences to generate analytics data.")
                return False
            
            print("[OK] Provider Statistics:")
            for stat in stats:
                print(f"\n  Provider: {stat['provider']}")
                print(f"    Count: {stat['count']}")
                print(f"    Avg Tokens: {stat['avg_tokens']:.2f}")
                print(f"    Avg Latency: {stat['avg_latency_ms']:.2f} ms")
                print(f"    Total Tokens: {stat['total_tokens']}")
            
            # Get total count
            total = repo.get_total_count()
            print(f"\n[OK] Total inferences in database: {total}")
            
            return True
            
    except Exception as e:
        print(f"[FAIL] Error checking analytics: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_inference_completeness():
    """Verify that stored inferences have all required fields."""
    print("\n" + "="*60)
    print("Verifying Inference Data Completeness")
    print("="*60)
    
    try:
        db = DatabaseConnection(config.database.connection_string)
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            
            # Get recent inferences
            recent = repo.query_inferences(limit=10)
            
            if not recent:
                print("[WARN] No inferences to verify")
                return False
            
            issues = []
            for inf in recent:
                if not inf.prompt:
                    issues.append(f"ID {inf.id}: Missing prompt")
                if not inf.response:
                    issues.append(f"ID {inf.id}: Missing response")
                if not inf.provider:
                    issues.append(f"ID {inf.id}: Missing provider")
                if not inf.created_at:
                    issues.append(f"ID {inf.id}: Missing created_at")
            
            if issues:
                print("[FAIL] Found data completeness issues:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print(f"[OK] All {len(recent)} recent inference(s) have complete data")
                return True
                
    except Exception as e:
        print(f"[FAIL] Error verifying completeness: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("End-to-End Test Verification")
    print("="*60)
    print()
    
    results = []
    
    # Check 1: Database initialization
    results.append(("Database Initialized", check_database_initialized()))
    
    # Check 2: Recent inferences
    results.append(("Recent Inferences", check_recent_inferences(minutes=10)))
    
    # Check 3: Analytics data
    results.append(("Analytics Data", check_analytics_data()))
    
    # Check 4: Data completeness
    results.append(("Data Completeness", verify_inference_completeness()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[SUCCESS] All checks passed! End-to-end test successful.")
        return 0
    else:
        print("[WARN] Some checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

