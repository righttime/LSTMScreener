#!/bin/bash
# =============================================================================
# daily_pipeline.sh — LSTMScreener 일일 자동화 파이프라인
# =============================================================================
# 각 단계를 서브셸로 분리하여 메모리 독립 실행
# 개별 단계 실패해도 다음 단계 진행 (단, 데이터/학습 실패는 중단)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TODAY="$(date +%Y%m%d)"
LOG_FILE="$LOG_DIR/pipeline_$TODAY.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

FAIL=0

# ──────────────────────────────────────────────────────────────────────────
log "============================================"
log "  [daily_pipeline.sh] 시작"
log "  날짜: $TODAY"
log "============================================"

# =============================================================================
# 1단계: 데이터 업데이트 (서브셸로 메모리 독립)
# =============================================================================
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  1️⃣  데이터 업데이트 (update_data.py)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ( python3 update_data.py >> "$LOG_FILE" 2>&1 ); then
    log "✅ 데이터 업데이트 완료"
else
    log "❌ 데이터 업데이트 실패. 파이프라인 중단."
    exit 1
fi

# =============================================================================
# 2단계: 추가 학습 (서브셸로 메모리 독립 — PyTorch 메모리 해제)
# =============================================================================
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  2️⃣  추가 학습 (retrain.py)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ( python3 retrain.py >> "$LOG_FILE" 2>&1 ); then
    log "✅ 추가 학습 완료"
else
    log "❌ 추가 학습 실패. 파이프라인 중단."
    exit 1
fi

# 메모리 정리
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
log "🗑️  캐시 메모리 정리 완료"

# =============================================================================
# 3단계: 순수 LSTM 스크리닝 (서브셸)
# =============================================================================
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  3️⃣  스크리닝 (screen_stocks.py)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RESULTS_FILE="$RESULTS_DIR/screen_$TODAY.txt"

if ( python3 screen_stocks.py > "$RESULTS_FILE" 2>&1 ); then
    log "✅ 스크리닝 완료: $RESULTS_FILE"
else
    log "⚠️  스크리닝 실패"
    FAIL=1
fi

# =============================================================================
# 3.5단계: 뉴스 감성 분석 스크리닝 (서브셸 — 실패해도 계속)
# =============================================================================
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  3.5  뉴스 분석 스크리닝 (screen_with_news.py)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ( python3 screen_with_news.py >> "$LOG_FILE" 2>&1 ); then
    log "✅ 뉴스 분석 스크리닝 완료"
else
    log "⚠️  뉴스 분석 스크리닝 실패 → 순수 LSTM 결과로 대체"
    FAIL=1
fi

# =============================================================================
# 4단계: LSTM 와치리스트 → TA 복사
# =============================================================================
SRC_WATCHLIST="$SCRIPT_DIR/data/watchlist.json"
DST_WATCHLIST="/root/project/TraderAlfred/data/watchlist.json"

if [ -f "$SRC_WATCHLIST" ]; then
    cp "$SRC_WATCHLIST" "$DST_WATCHLIST"
    log "✅ LSTM 와치리스트 → TraderAlfred 복사 완료"
else
    log "⚠️  LSTM 와치리스트 없음"
    FAIL=1
fi

# ──────────────────────────────────────────────────────────────────────────
log ""
log "============================================"
if [ $FAIL -eq 0 ]; then
    log "  ✅ 파이프라인 완료!"
else
    log "  ⚠️  파이프라인 부분 완료 (일부 단계 실패)"
fi
log "  결과 파일: $RESULTS_FILE"
log "  로그 파일: $LOG_FILE"
log "============================================"

# ──────────────────────────────────────────────────────────────────────────
# 텔레그램 알림: 학습 완료 보고
# ──────────────────────────────────────────────────────────────────────────
source /root/project/TraderAlfred/.env 2>/dev/null
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    TRAIN_TIME=$(grep "학습 시간:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP "[0-9.]+초" | head -1)
    TRAIN_MIN=""
    if [ -n "$TRAIN_TIME" ]; then
        SECS=$(echo "$TRAIN_TIME" | sed 's/초//')
        TRAIN_MIN=$(echo "$SECS / 60" | awk "{printf "%.0f", $1}")
    fi
    MSG="🧠 *LSTM 추가 학습 완료!*\n\n⏱ 소요시간: ${TRAIN_MIN:-?}분\n📈 스크리닝: $RESULTS_FILE"
    curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d "chat_id=$TELEGRAM_CHAT_ID" \
        -d "text=$MSG" \
        -d "parse_mode=Markdown" > /dev/null 2>&1
    log "📱 텔레그램 알림 전송 완료"
fi
