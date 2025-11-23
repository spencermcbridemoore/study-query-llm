# End-to-End Testing Checklist

## Pre-Test Setup ✓
- [x] Database initialized
- [x] .env file configured with Azure deployment (gpt-4o)
- [x] Verification script created

## Test 1: GUI Startup
- [ ] GUI loads without errors
- [ ] Both tabs visible (Inference and Analytics)
- [ ] No console errors

## Test 2: Inference Flow
- [ ] Select Azure provider from dropdown
- [ ] Click "Load Deployments" button
- [ ] Verify "gpt-4o" appears in deployment dropdown
- [ ] Enter test prompt: "Say hello in one word"
- [ ] Set temperature (optional)
- [ ] Set max tokens (optional)
- [ ] Click "Run Inference" button
- [ ] Verify response appears in response area
- [ ] Verify metadata displays (tokens, latency)
- [ ] Verify status message shows success

## Test 3: Database Storage Verification
Run verification script after inference:
```bash
python test_e2e_verification.py
```
- [ ] New inference appears in database
- [ ] All fields populated correctly
- [ ] Provider name is correct
- [ ] Timestamp is recent

## Test 4: Analytics Dashboard
- [ ] Switch to Analytics tab
- [ ] Verify summary statistics display
  - [ ] Total inferences count
  - [ ] Total tokens
  - [ ] Unique providers
- [ ] Verify provider comparison table shows data
- [ ] Verify recent inferences table shows new inference
- [ ] Click "Refresh Data" button
- [ ] Verify data updates

## Test 5: Multiple Inferences
- [ ] Run 2-3 more inferences with different prompts
- [ ] Verify all appear in database
- [ ] Verify analytics update correctly
- [ ] Verify recent inferences table updates

## Test 6: Error Handling
- [ ] Test with empty prompt (should show error)
- [ ] Test with invalid deployment (if possible)
- [ ] Verify error messages display properly
- [ ] Verify app doesn't crash on errors

## Test 7: Final Verification
Run verification script again:
```bash
python test_e2e_verification.py
```
- [ ] All checks pass
- [ ] Recent inferences found
- [ ] Analytics data correct
- [ ] Data completeness verified

## Notes
- Test results:
- Issues found:
- Fixes needed:

