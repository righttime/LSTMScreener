"""
일일 데이터 업데이트 스크립트
- FinanceDataReader로 최근 1일(또는 누락분) 캔들 데이터 수집
- market.db에 추가 (중복 날짜 건너뛰기, INSERT OR REPLACE)
- 전종목 대상
- 완료 후 로그 출력 (新增 N건, 更新 M건)
"""
import os, sys, sqlite3
from datetime import datetime, timedelta
import pandas as pd

# ── FinanceDataReader import (runtime check) ──────────────────────────────
try:
    import FinanceDataReader as fdr
except ImportError:
    print("[ERROR] FinanceDataReader가 설치되지 않았습니다.")
    print("  pip install finance-datareader")
    sys.exit(1)

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB     = os.path.join(PROJECT_DIR, "data", "market.db")

def get_latest_dates(conn):
    """DB에 저장된 각 종목의 마지막 날짜를 반환"""
    cur = conn.execute("SELECT code, MAX(date) as max_date FROM candles GROUP BY code")
    return {row[0]: row[1] for row in cur.fetchall()}

def get_all_codes(conn):
    """stocks 테이블의 모든 code 반환"""
    cur = conn.execute("SELECT code FROM stocks")
    return [row[0] for row in cur.fetchall()]

def upsert_candles(conn, rows):
    """candles 테이블에 INSERT OR REPLACE (중복이면 UPDATE)"""
    if not rows:
        return 0
    conn.executemany(
        """INSERT OR REPLACE INTO candles
              (code, date, open, high, low, close, volume, change_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows
    )
    conn.commit()
    return len(rows)

def to_fdr_date(s):
    """YYYYMMDD → YYYY-MM-DD"""
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

def fetch_and_upsert(conn, code, start_date_str, end_date_str):
    """FinanceDataReader에서 데이터 가져와서 upsert. 반환: (new_cnt, upd_cnt)"""
    try:
        df = fdr.DataReader(code, to_fdr_date(start_date_str), to_fdr_date(end_date_str))
    except Exception as e:
        return 0, 0

    if df.empty:
        return 0, 0

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    rows = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y%m%d")
        open_p   = int(row.get("Open",   0) or 0)
        high_p   = int(row.get("High",   0) or 0)
        low_p    = int(row.get("Low",    0) or 0)
        close_p  = int(row.get("Close",  0) or 0)
        vol      = int(row.get("Volume", 0) or 0)
        chg      = float(row.get("Change", 0) or 0)
        if close_p <= 0:
            continue
        rows.append((code, date_str, open_p, high_p, low_p, close_p, vol, chg))

    if not rows:
        return 0, 0

    # 기존 데이터 확인 (range check for efficiency)
    existing = set()
    cur = conn.execute(
        "SELECT date FROM candles WHERE code = ? AND date BETWEEN ? AND ?",
        (code, rows[0][1], rows[-1][1])
    )
    existing = {r[0] for r in cur.fetchall()}

    new_rows     = [r for r in rows if r[1] not in existing]
    updated_rows = [r for r in rows if r[1] in existing]

    count_new = upsert_candles(conn, new_rows)
    return count_new, len(updated_rows)

def main():
    print("=" * 60)
    print("  [update_data.py] 일일 데이터 업데이트")
    print(f"  시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    conn = sqlite3.connect(DATA_DB)
    conn.execute("PRAGMA journal_mode=WAL")

    latest_dates = get_latest_dates(conn)
    all_codes    = get_all_codes(conn)

    today_str     = datetime.now().strftime("%Y%m%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    if not latest_dates:
        start_str = "20210101"
    else:
        oldest_last = min(latest_dates.values())
        start_str = (datetime.strptime(oldest_last, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    # 오늘까지 Fetch (FDR이 빈 데이터 반환하면 무시됨)
    end_str = today_str

    print(f"\n📦 전체 종목: {len(all_codes)}개")
    print(f"📅 Fetch 범위: {start_str} ~ {end_str}")
    print(f"🔍 누락 확인 중...\n")

    total_new = 0
    total_upd = 0
    fail_codes = []
    skipped = 0

    for i, code in enumerate(all_codes, 1):
        db_last = latest_dates.get(code, "20210101")
        code_start = (datetime.strptime(db_last, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

        # 이미 최신이면 스킵
        if code_start > end_str:
            skipped += 1
            continue

        new_cnt, upd_cnt = fetch_and_upsert(conn, code, code_start, end_str)

        if new_cnt > 0 or upd_cnt > 0:
            total_new += new_cnt
            total_upd += upd_cnt
        elif new_cnt == 0 and upd_cnt == 0:
            # FDR이 빈 데이터(공휴일 등)를 반환한 경우 — fail_codes에 추가하지 않음
            pass

        if (i % 50 == 0) or (i == len(all_codes)):
            print(f"\r  [{i}/{len(all_codes)}] 진행중...  新增:{total_new}  更新:{total_upd}", end="", flush=True)

    conn.close()

    print(f"\n\n{'='*60}")
    print(f"  ✅ 업데이트 완료!")
    print(f"     新增 (INSERT): {total_new}건")
    print(f"     更新 (UPDATE): {total_upd}건")
    print(f"     이미 최신(skip): {skipped}개")
    print(f"     실패 종목: {len(fail_codes)}개")
    if fail_codes:
        fail_list = ', '.join(fail_codes[:10])
        fail_suffix = '...' if len(fail_codes) > 10 else ''
        print(f"     실패 목록: {fail_list}{fail_suffix}")
    print(f"{'='*60}")

    # DB 최종 날짜 확인
    conn2 = sqlite3.connect(DATA_DB)
    cur = conn2.execute(
        "SELECT MIN(max_d), MAX(max_d) FROM (SELECT MAX(date) as max_d FROM candles GROUP BY code)"
    )
    min_d, max_d = cur.fetchone()
    print(f"  DB 날짜 범위: {min_d} ~ {max_d}")
    conn2.close()

if __name__ == "__main__":
    main()
