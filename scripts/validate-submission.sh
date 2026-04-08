#!/usr/bin/env bash
# scripts/validate-submission.sh
#
# Smoke-test the environment server end-to-end.
# All three tasks are exercised plus edge-cases (redundancy, invalid action).
#
# Usage:
#   bash scripts/validate-submission.sh [SERVER_URL]
#   Default SERVER_URL: http://localhost:7860

set -euo pipefail

SERVER="${1:-http://localhost:7860}"
PASS=0
FAIL=0

green()  { echo -e "\033[0;32m[PASS] $*\033[0m"; }
red()    { echo -e "\033[0;31m[FAIL] $*\033[0m"; }
header() { echo -e "\n\033[1;34m=== $* ===\033[0m"; }

check() {
  local label="$1" result="$2"
  if [ "$result" = "true" ]; then
    green "$label"; PASS=$((PASS+1))
  else
    red   "$label"; FAIL=$((FAIL+1))
  fi
}

# Extract a top-level JSON field as a raw lowercase string.
#   null  -> "null"
#   true  -> "true"    false -> "false"
#   other -> the string value
jget() {
  local json="$1" key="$2"
  echo "$json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
v = d.get('$key')
if v is None:          print('null')
elif isinstance(v,bool): print(str(v).lower())
else:                  print(v)
" 2>/dev/null || echo ""
}

# Extract reward.total (nested)
rtotal() {
  echo "$1" | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d.get('reward',{}).get('total',0))
" 2>/dev/null || echo 0
}

pos()  { python3 -c "print('true' if float('$1') >  0    else 'false')"; }
neg()  { python3 -c "print('true' if float('$1') <  0    else 'false')"; }
gte()  { python3 -c "print('true' if float('$1') >= float('$2') else 'false')"; }
is4xx(){ python3 -c "print('true' if 400 <= int('$1') < 500 else 'false')"; }

# ---------------------------------------------------------------------------
header "Health check"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER/health")
check "GET /health returns 200" "$([ "$CODE" = "200" ] && echo true || echo false)"

# ---------------------------------------------------------------------------
header "Easy task: reset"
R=$(curl -s -X POST "$SERVER/reset" -H "Content-Type: application/json" -d '{"difficulty":"easy"}')
check "reset() returns invoice_id"        "$(V=$(jget "$R" invoice_id); [ -n "$V" ] && [ "$V" != "null" ] && echo true || echo false)"
check "Easy task: gst_number is null"     "$([ "$(jget "$R" gst_number)" = "null" ] && echo true || echo false)"

# ---------------------------------------------------------------------------
header "Easy task: add_issue"
S=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
    -d '{"action_type":"add_issue","value":"missing_gst_number"}')
check "add_issue(missing_gst_number) → +reward" "$(pos "$(rtotal "$S")")"

# ---------------------------------------------------------------------------
header "Easy task: finalize"
curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
     -d '{"action_type":"request_missing_info"}' > /dev/null
F=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
    -d '{"action_type":"finalize_review"}')
check "finalize_review sets done=true" "$([ "$(jget "$F" done)" = "true" ] && echo true || echo false)"
check "Easy final score > 0"           "$(pos "$(rtotal "$F")")"

# ---------------------------------------------------------------------------
header "Medium task: wrong total"
curl -s -X POST "$SERVER/reset" -H "Content-Type: application/json" -d '{"difficulty":"medium"}' > /dev/null
S2=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
     -d '{"action_type":"add_issue","value":"wrong_total_calculation"}')
check "detect wrong_total_calculation → +reward" "$(pos "$(rtotal "$S2")")"
curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" -d '{"action_type":"mark_invalid"}' > /dev/null
FM=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" -d '{"action_type":"finalize_review"}')
check "Medium final score > 0" "$(pos "$(rtotal "$FM")")"

# ---------------------------------------------------------------------------
header "Hard task: five issues"
HR=$(curl -s -X POST "$SERVER/reset" -H "Content-Type: application/json" -d '{"difficulty":"hard"}')
check "Hard: duplicate_flag=true"    "$([ "$(jget "$HR" duplicate_flag)"  = "true"  ] && echo true || echo false)"
check "Hard: receipt_attached=false" "$([ "$(jget "$HR" receipt_attached)" = "false" ] && echo true || echo false)"
check "Hard: vendor_name=null"       "$([ "$(jget "$HR" vendor_name)"     = "null"  ] && echo true || echo false)"
check "Hard: gst_number=null"        "$([ "$(jget "$HR" gst_number)"      = "null"  ] && echo true || echo false)"

for ISS in missing_vendor_name missing_gst_number wrong_total_calculation duplicate_invoice missing_receipt; do
  SR=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
       -d "{\"action_type\":\"add_issue\",\"value\":\"$ISS\"}")
  check "add_issue($ISS) → +reward" "$(pos "$(rtotal "$SR")")"
done

curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" -d '{"action_type":"flag_duplicate"}' > /dev/null
curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" -d '{"action_type":"mark_invalid"}'   > /dev/null
HE=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" -d '{"action_type":"finalize_review"}')
check "Hard done=true"          "$([ "$(jget "$HE" done)" = "true" ] && echo true || echo false)"
check "Hard score >= 0.5"       "$(gte "$(rtotal "$HE")" 0.5)"

# ---------------------------------------------------------------------------
header "Redundancy and false-positive penalties"
curl -s -X POST "$SERVER/reset" -H "Content-Type: application/json" -d '{"difficulty":"easy"}' > /dev/null
curl -s -X POST "$SERVER/step"  -H "Content-Type: application/json" \
     -d '{"action_type":"add_issue","value":"missing_gst_number"}' > /dev/null

DUP=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
      -d '{"action_type":"add_issue","value":"missing_gst_number"}')
check "Duplicate add_issue → negative reward" "$(neg "$(rtotal "$DUP")")"

FP=$(curl -s -X POST "$SERVER/step" -H "Content-Type: application/json" \
     -d '{"action_type":"add_issue","value":"invented_issue_xyz"}')
check "False-positive issue → negative reward" "$(neg "$(rtotal "$FP")")"

# ---------------------------------------------------------------------------
header "Invalid action → 4xx"
BAD=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVER/step" \
      -H "Content-Type: application/json" -d '{"action_type":"does_not_exist"}')
check "Invalid action_type returns 4xx" "$(is4xx "$BAD")"

# ---------------------------------------------------------------------------
header "GET /state"
curl -s -X POST "$SERVER/reset" -H "Content-Type: application/json" -d '{"difficulty":"medium"}' > /dev/null
SC=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER/state")
check "GET /state returns 200" "$([ "$SC" = "200" ] && echo true || echo false)"

# ---------------------------------------------------------------------------
header "Summary"
TOTAL=$((PASS+FAIL))
echo ""
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
  red "$FAIL test(s) failed."
  exit 1
else
  green "All $PASS checks passed!"
fi
