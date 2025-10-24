# Testing Checklist - Human Purchase Intent Web App

## Prerequisites

Before testing, ensure:
- [ ] FastAPI backend is running on `http://localhost:8000`
- [ ] Backend has CORS configured for `http://localhost:3000`
- [ ] `.env.local` is configured with `NEXT_PUBLIC_API_URL=http://localhost:8000`
- [ ] Dependencies are installed (`npm install`)
- [ ] Development server is running (`npm run dev`)

## End-to-End Test Workflow

### 1. Dashboard (Home Page)

**URL:** `http://localhost:3000/`

**Tests:**
- [ ] Page loads without errors
- [ ] "New Survey" button is visible in header
- [ ] Empty state shows when no surveys exist
- [ ] Loading spinner appears during data fetch
- [ ] Error message displays if API is unreachable

**Expected Behavior:**
- If no surveys: Shows empty state with "Create your first survey" message
- If surveys exist: Displays surveys grouped by status (In Progress, Completed, Pending)

---

### 2. Create Survey Flow

**URL:** `http://localhost:3000/surveys/new`

#### Step 1: Product Details
- [ ] Product name input accepts text
- [ ] Product description textarea accepts text
- [ ] Validation errors show for empty fields
- [ ] Validation errors show for description < 10 characters
- [ ] "Next" button navigates to Step 2

#### Step 2: Cohort Settings
- [ ] Cohort size input accepts numbers
- [ ] Validation enforces min: 10, max: 1000
- [ ] Help text displays correctly
- [ ] "Previous" button returns to Step 1
- [ ] "Next" button navigates to Step 3

#### Step 3: Demographics
- [ ] "Enable Demographic Targeting" toggle works
- [ ] Demographic fields appear when toggle is ON
- [ ] Gender dropdown has options: Any, Male, Female, Other
- [ ] Income dropdown has options: Any, Low, Middle, High
- [ ] Location input accepts text (optional)
- [ ] "Previous" button returns to Step 2
- [ ] "Create Survey" button submits form

#### Submit & Redirect
- [ ] Loading spinner shows during submission
- [ ] On success: Redirects to survey results page (`/surveys/{id}`)
- [ ] On error: Shows error message
- [ ] Form data is properly sent to backend

**Test Data:**
```
Product Name: Test Product A
Description: A premium eco-friendly water bottle with advanced insulation
Cohort Size: 100
Demographics: Enabled
  - Gender: Any
  - Income: Middle
  - Location: United States
```

---

### 3. Survey Results Page

**URL:** `http://localhost:3000/surveys/{id}` (auto-redirect from create)

#### Running State
- [ ] Shows "Survey in Progress" message
- [ ] Animated pulse icon displays
- [ ] Status badge shows "Running"
- [ ] Page polls every 3 seconds for status updates
- [ ] No results shown while running

#### Completed State
- [ ] SSR Rating badge displays with correct color
- [ ] Rating is between 1-5
- [ ] Confidence percentage shows (e.g., "85.3%")
- [ ] Mean rating displays (e.g., "3.45")
- [ ] Distribution chart renders with 5 bars
- [ ] Chart bars are color-coded (red to green)
- [ ] Survey metadata shows:
  - [ ] Created date (relative time)
  - [ ] Cohort size
  - [ ] Demographic targeting badge (if enabled)
- [ ] Target demographics display (if enabled)
- [ ] "Back to Dashboard" button works

#### Failed State
- [ ] Red error card displays
- [ ] Error message explains the failure
- [ ] No distribution chart shown

**Navigation:**
- [ ] "Back to Dashboard" button returns to `/`

---

### 4. Compare Surveys

**URL:** `http://localhost:3000/compare`

#### Empty State
- [ ] Shows "No Surveys Selected" message
- [ ] Prompts to select at least 2 surveys

#### Survey Selection
- [ ] Dropdown shows completed surveys only
- [ ] Can select multiple surveys (2-4)
- [ ] Selected surveys display as tags with remove (X) buttons
- [ ] URL updates with query params: `?ids=id1,id2,id3`
- [ ] Page state persists on refresh (reads from URL)

#### Comparison View (2+ surveys selected)
- [ ] SSR Rating Cards section displays
  - [ ] Each survey has its own card
  - [ ] Rating badge shows correctly
  - [ ] Confidence percentage displays
  - [ ] Mean rating displays
  - [ ] Cohort size shows
- [ ] Distribution Charts section displays
  - [ ] Each survey has its own chart
  - [ ] Product name shows above each chart
  - [ ] Charts are stacked vertically
- [ ] Detailed Metrics Table displays
  - [ ] SSR Rating row
  - [ ] Confidence row
  - [ ] Mean Rating row
  - [ ] Std Dev row
  - [ ] Data aligns correctly in columns

**Test Workflow:**
1. Create 2-3 test surveys with different products
2. Wait for all to complete
3. Navigate to Compare page
4. Select surveys from dropdown
5. Verify all comparison views render correctly

---

## API Integration Tests

### Health Check
- [ ] Backend is running: `curl http://localhost:8000/health` (if endpoint exists)
- [ ] CORS headers present in responses

### Survey Creation
```bash
# Should return 201 Created with survey object
curl -X POST http://localhost:8000/api/v1/surveys \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Test Product",
    "product_description": "A test product for purchase intent analysis",
    "cohort_size": 100,
    "use_demographics": false
  }'
```

### Survey List
```bash
# Should return array of surveys
curl http://localhost:8000/api/v1/surveys
```

### Survey Results
```bash
# Replace {id} with actual survey ID
curl http://localhost:8000/api/v1/surveys/{id}/results
```

---

## Visual/UI Tests

### Responsive Design
- [ ] Dashboard works on mobile (375px width)
- [ ] Survey form is usable on tablet (768px width)
- [ ] Compare page is readable on mobile
- [ ] All buttons are tappable on touch devices

### Accessibility
- [ ] All forms have proper labels
- [ ] Error messages are announced
- [ ] Keyboard navigation works (Tab key)
- [ ] Focus states are visible

### Loading States
- [ ] Spinners show during data fetching
- [ ] Button text changes to "Loading..." during submission
- [ ] Disabled states prevent double-submission

### Error Handling
- [ ] Network errors show user-friendly messages
- [ ] 404 errors show "Survey not found"
- [ ] Validation errors are clear and helpful

---

## Performance Tests

- [ ] Initial page load < 3 seconds
- [ ] Navigation between pages is instant
- [ ] Chart rendering is smooth (no lag)
- [ ] Polling doesn't cause UI freezes

---

## Browser Compatibility

Test in multiple browsers:
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (macOS/iOS)
- [ ] Mobile browsers (iOS Safari, Chrome Android)

---

## Common Issues & Solutions

### Issue: CORS Error
**Symptom:** "Access to fetch blocked by CORS policy"
**Solution:** Verify FastAPI CORS middleware allows `http://localhost:3000`

### Issue: API Connection Failed
**Symptom:** "Network error: Unable to reach API server"
**Solution:**
1. Check FastAPI is running on port 8000
2. Verify `NEXT_PUBLIC_API_URL` in `.env.local`
3. Check firewall isn't blocking connections

### Issue: Surveys Not Loading
**Symptom:** Empty dashboard despite surveys existing
**Solution:**
1. Check browser console for errors
2. Verify API returns data: `curl http://localhost:8000/api/v1/surveys`
3. Check React Query DevTools for query status

### Issue: Polling Not Working
**Symptom:** Running survey status never updates
**Solution:**
1. Verify backend task is actually running
2. Check polling interval in browser Network tab
3. Look for errors in console

---

## Automated Testing (Future)

Future improvements:
- [ ] Unit tests with Jest/Vitest
- [ ] Component tests with React Testing Library
- [ ] E2E tests with Playwright
- [ ] API integration tests
- [ ] Visual regression tests

---

## Test Report Template

After completing tests, fill out this report:

**Date:** _____________
**Tester:** _____________
**Environment:**
- Frontend Version: _____________
- Backend Version: _____________
- Node.js Version: _____________

**Results:**
- Total Tests: _____
- Passed: _____
- Failed: _____
- Blocked: _____

**Failed Tests:**
1. _____________________________________________
2. _____________________________________________

**Notes:**
___________________________________________________
___________________________________________________
