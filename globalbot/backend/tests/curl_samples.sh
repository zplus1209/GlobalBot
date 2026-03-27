#!/usr/bin/env bash
# ─── Manual curl test samples ────────────────────────────────────────────────
# Start server first:
#   python serve.py --mode online --model_name gemini --model_version gemini-1.5-flash
#   (or offline: --mode offline --model_engine ollama --model_version llama3.2)
#
# Run all: bash tests/curl_samples.sh
# Run one: bash tests/curl_samples.sh test_upload

BASE="http://localhost:5002"
DOC_ID=""   # filled after upload


check() { echo; echo "=== $1 ==="; }
ok()    { echo "[PASS] $1"; }
fail()  { echo "[FAIL] $1"; exit 1; }
jq_or_cat() { command -v jq &>/dev/null && jq . || cat; }


# ─── 1. List documents (empty) ───────────────────────────────────────────────
test_list_empty() {
  check "1. List documents"
  RES=$(curl -sf "$BASE/api/documents")
  echo "$RES" | jq_or_cat
  [[ "$RES" == "[]" || "$RES" == *"doc_id"* ]] && ok "list" || fail "list"
}

# ─── 2. Upload PDF ───────────────────────────────────────────────────────────
test_upload() {
  check "2. Upload PDF"
  # Create minimal test PDF if none exists
  if [ ! -f /tmp/test_sample.pdf ]; then
    python3 - <<'PY'
import struct
pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 80>>
stream
BT /F1 12 Tf 72 720 Td (Aviation Fuel JIG Standard - Quality Control Section) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000404 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
467
%%EOF"""
open("/tmp/test_sample.pdf", "wb").write(pdf)
print("Created /tmp/test_sample.pdf")
PY
  fi

  RES=$(curl -sf -X POST "$BASE/api/documents" \
    -F "file=@/tmp/test_sample.pdf;type=application/pdf")
  echo "$RES" | jq_or_cat
  DOC_ID=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['doc_id'])" 2>/dev/null)
  export DOC_ID
  echo "Uploaded doc_id: $DOC_ID"
  [ -n "$DOC_ID" ] && ok "upload" || fail "upload"
}

# ─── 3. Poll status ──────────────────────────────────────────────────────────
test_poll_status() {
  check "3. Poll processing status"
  for i in $(seq 1 20); do
    STATUS=$(curl -sf "$BASE/api/documents/$DOC_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "  attempt $i: $STATUS"
    [ "$STATUS" = "ready" ] && { ok "processing completed"; return; }
    [ "$STATUS" = "error" ] && fail "processing returned error"
    sleep 5
  done
  fail "timeout waiting for ready"
}

# ─── 4. Get document info ────────────────────────────────────────────────────
test_get_doc() {
  check "4. Get document"
  curl -sf "$BASE/api/documents/$DOC_ID" | jq_or_cat
  ok "get doc"
}

# ─── 5. Get blocks ───────────────────────────────────────────────────────────
test_get_blocks() {
  check "5. Get extracted blocks"
  RES=$(curl -sf "$BASE/api/documents/$DOC_ID/blocks")
  COUNT=$(echo "$RES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
  echo "  blocks: $COUNT"
  echo "$RES" | python3 -c "import sys,json; blocks=json.load(sys.stdin); [print(f'  [{b[\"label\"]:12}] page={b[\"page\"]} bbox={b[\"bbox\"]}') for b in blocks[:5]]" 2>/dev/null
  ok "get blocks"
}

# ─── 6. Serve file ───────────────────────────────────────────────────────────
test_serve_file() {
  check "6. Serve original file"
  HTTP=$(curl -so /dev/null -w "%{http_code}" "$BASE/api/documents/$DOC_ID/file")
  CT=$(curl -sI "$BASE/api/documents/$DOC_ID/file" | grep -i content-type | head -1)
  echo "  HTTP: $HTTP | $CT"
  [ "$HTTP" = "200" ] && ok "serve file" || fail "serve file (HTTP $HTTP)"
}

# ─── 7. Document Q&A ─────────────────────────────────────────────────────────
test_doc_chat() {
  check "7. Document Q&A - ask about content"
  RES=$(curl -sf -X POST "$BASE/api/chat/document" \
    -H "Content-Type: application/json" \
    -d "{
      \"doc_id\": \"$DOC_ID\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What are the main topics in this document?\"}],
      \"k\": 3
    }")
  echo "$RES" | jq_or_cat
  ANSWER=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][:200])" 2>/dev/null)
  NDOCS=$(echo "$RES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['retrieved_docs']))" 2>/dev/null)
  echo ""
  echo "  answer (truncated): $ANSWER"
  echo "  retrieved_docs: $NDOCS"
  [ -n "$ANSWER" ] && ok "doc chat" || fail "doc chat"
}

# ─── 8. Check retrieved_docs have bbox + page ────────────────────────────────
test_bbox_in_response() {
  check "8. retrieved_docs bbox structure"
  RES=$(curl -sf -X POST "$BASE/api/chat/document" \
    -H "Content-Type: application/json" \
    -d "{
      \"doc_id\": \"$DOC_ID\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What standards are referenced?\"}],
      \"k\": 5
    }")
  python3 - <<PY
import json, sys
data = json.loads("""$RES""")
for i, d in enumerate(data.get("retrieved_docs", [])[:3]):
    print(f"  source {i+1}:")
    print(f"    page     = {d['page']}")
    print(f"    bbox     = {d['bbox']}")
    print(f"    label    = {d['label']}")
    print(f"    score    = {d['score']:.3f}")
    print(f"    doc_id   = {d['doc_id'][:12]}...")
    assert isinstance(d["bbox"], list), "bbox must be a list"
    assert d["page"] > 0, "page must be > 0"
    assert 0 <= d["score"] <= 1, "score must be 0-1"
print("[PASS] bbox structure valid")
PY
}

# ─── 9. Knowledge base chat ───────────────────────────────────────────────────
test_kb_chat() {
  check "9. Knowledge base chat"
  RES=$(curl -sf -X POST "$BASE/api/chat/knowledge" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Summarize what you know about fuel quality standards."}],
      "k": 3
    }')
  echo "$RES" | jq_or_cat
  ok "knowledge chat"
}

# ─── 10. Multi-turn chat ─────────────────────────────────────────────────────
test_multi_turn() {
  check "10. Multi-turn document chat"
  RES=$(curl -sf -X POST "$BASE/api/chat/document" \
    -H "Content-Type: application/json" \
    -d "{
      \"doc_id\": \"$DOC_ID\",
      \"messages\": [
        {\"role\": \"user\",      \"content\": \"What is this document?\"},
        {\"role\": \"assistant\", \"content\": \"It covers aviation fuel standards.\"},
        {\"role\": \"user\",      \"content\": \"What specific sections are mentioned?\"}
      ],
      \"k\": 3
    }")
  ANSWER=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][:200])" 2>/dev/null)
  echo "  follow-up answer: $ANSWER"
  [ -n "$ANSWER" ] && ok "multi-turn" || fail "multi-turn"
}

# ─── 11. Error cases ─────────────────────────────────────────────────────────
test_errors() {
  check "11. Error handling"
  H=$(curl -so /dev/null -w "%{http_code}" -X POST "$BASE/api/documents" -F "file=@/dev/null;type=text/csv" 2>/dev/null)
  echo "  upload bad type: HTTP $H"; [ "$H" = "400" ] && ok "400 on bad type" || fail "expected 400"

  H=$(curl -so /dev/null -w "%{http_code}" "$BASE/api/documents/nonexistent" 2>/dev/null)
  echo "  get nonexistent: HTTP $H"; [ "$H" = "404" ] && ok "404 on nonexistent" || fail "expected 404"
}

# ─── 12. Delete ──────────────────────────────────────────────────────────────
test_delete() {
  check "12. Delete document"
  H=$(curl -so /dev/null -w "%{http_code}" -X DELETE "$BASE/api/documents/$DOC_ID")
  echo "  HTTP: $H"
  [ "$H" = "204" ] && ok "delete" || fail "delete (HTTP $H)"
  H2=$(curl -so /dev/null -w "%{http_code}" "$BASE/api/documents/$DOC_ID")
  [ "$H2" = "404" ] && ok "gone after delete" || fail "still exists after delete"
}


# ─── Runner ───────────────────────────────────────────────────────────────────
RUN="${1:-all}"

if [ "$RUN" = "all" ]; then
  test_list_empty
  test_upload
  test_poll_status
  test_get_doc
  test_get_blocks
  test_serve_file
  test_doc_chat
  test_bbox_in_response
  test_kb_chat
  test_multi_turn
  test_errors
  test_delete
  echo; echo "=== All tests done ==="
else
  $RUN
fi
