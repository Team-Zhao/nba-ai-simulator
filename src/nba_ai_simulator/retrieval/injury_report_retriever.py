from pathlib import Path
from datetime import datetime, timedelta
from pypdf import PdfReader
import pdfplumber
import requests
import pandas as pd
import re
import unicodedata
from nba_api.stats.static import players
from nba_ai_simulator.agents.availability_agent import (
    AvailabilityUpdate,
    validate_availability_update,
)

NBA_INJURY_REPORT_BASE_URL = (
    "https://ak-static.cms.nba.com/referee/injury"
)

RAW_INJURY_DIR = Path(
    "data/raw/injury_reports"
)

PROCESSED_INJURY_DIR = Path(
    "data/processed/injury_reports"
)

def build_injury_report_dataframe(pdf_path):
    rows = extract_report_rows(pdf_path)

    df = pd.DataFrame(rows)

    report_published_at = (
        extract_report_published_at(
            pdf_path
        )
    )

    df.insert(
        0,
        "reportPublishedAt",
        report_published_at,
    )

    return df

def save_injury_report_dataframe(
    df,
    report_date,
    report_hour,
):
    PROCESSED_INJURY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_INJURY_DIR
        / f"injury_report_{report_date:%Y-%m-%d}_{report_hour}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path

def extract_report_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text(
            extraction_mode="layout"
        )

        if text:
            pages.append(text)

    return "\n".join(pages)

def inspect_report_tables(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            print(
                f"\nPage {page_number}: "
                f"{len(tables)} tables found"
            )

            for table_number, table in enumerate(
                tables,
                start=1,
            ):
                print(
                    f"\nTable {table_number}: "
                    f"{len(table)} rows"
                )

                for row in table[:5]:
                    print(row)

def extract_player_rows_from_page(page):
    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
    )

    player_words = [
        word
        for word in words
        if 400 <= word["x0"] < 560
        and word["top"] > 120
        and word["text"] != "PlayerName"
    ]

    status_words = [
        word
        for word in words
        if 560 <= word["x0"] < 650
        and word["top"] > 120
        and word["text"] != "CurrentStatus"
    ]

    reason_words = [
        word
        for word in words
        if word["x0"] >= 650
        and word["top"] > 120
        and word["text"] != "Reason"
    ]

    rows = []

    for i, player in enumerate(player_words):
        player_top = player["top"]

        if i == 0:
            upper_bound = 120
        else:
            upper_bound = (
                player_words[i - 1]["top"]
                + player_top
            ) / 2

        if i == len(player_words) - 1:
            lower_bound = player_top + 20
        else:
            lower_bound = (
                player_top
                + player_words[i + 1]["top"]
            ) / 2

        status_matches = [
            word["text"]
            for word in status_words
            if upper_bound <= word["top"] < lower_bound
        ]

        reason_matches = [
            word["text"]
            for word in reason_words
            if upper_bound <= word["top"] < lower_bound
        ]

        rows.append(
            {
                "playerName": player["text"],
                "status": " ".join(status_matches),
                "reason": " ".join(reason_matches),
            }
        )

    return rows

def build_master_player_lookup():
    all_players = players.get_players()

    rows = []

    for player in all_players:
        rows.append(
            {
                "personId": player["id"],
                "firstName": player["first_name"],
                "familyName": player["last_name"],
                "fullName": player["full_name"],
                "isActive": player["is_active"],
            }
        )

    lookup = pd.DataFrame(rows)

    lookup["normalizedName"] = (
        lookup["fullName"]
        .apply(normalize_name)
    )

    return lookup

def extract_rows_from_page(
    page,
    state=None,
):
    if state is None:
        state = {
            "gameDate": None,
            "gameTime": None,
            "matchup": None,
            "team": None,
        }
    words = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
    )

    # Ignore report header / column header area
    words = [
        word
        for word in words
        if word["top"] > 120
    ]

    game_date_words = [
        word
        for word in words
        if 0 <= word["x0"] < 100
    ]

    game_time_words = [
        word
        for word in words
        if 100 <= word["x0"] < 180
    ]

    matchup_words = [
        word
        for word in words
        if 180 <= word["x0"] < 250
    ]

    team_words = [
        word
        for word in words
        if 250 <= word["x0"] < 400
    ]

    player_words = [
        word
        for word in words
        if 400 <= word["x0"] < 560
    ]

    status_words = [
        word
        for word in words
        if 560 <= word["x0"] < 650
    ]

    reason_words = [
        word
        for word in words
        if word["x0"] >= 650
    ]

    rows = []

    current_game_date = state["gameDate"]
    current_game_time = state["gameTime"]
    current_matchup = state["matchup"]
    current_team = state["team"]

    for i, player in enumerate(player_words):
        player_top = player["top"]

        if i == 0:
            upper_bound = 120
        else:
            upper_bound = (
                player_words[i - 1]["top"]
                + player_top
            ) / 2

        if i == len(player_words) - 1:
            lower_bound = player_top + 20
        else:
            lower_bound = (
                player_top
                + player_words[i + 1]["top"]
            ) / 2

        game_date_matches = [
            word["text"]
            for word in game_date_words
            if upper_bound <= word["top"] < lower_bound
        ]

        game_time_matches = [
            word["text"]
            for word in game_time_words
            if upper_bound <= word["top"] < lower_bound
        ]

        matchup_matches = [
            word["text"]
            for word in matchup_words
            if upper_bound <= word["top"] < lower_bound
        ]

        team_matches = [
            word["text"]
            for word in team_words
            if upper_bound <= word["top"] < lower_bound
        ]

        status_matches = [
            word["text"]
            for word in status_words
            if upper_bound <= word["top"] < lower_bound
        ]

        reason_matches = [
            word["text"]
            for word in reason_words
            if upper_bound <= word["top"] < lower_bound
        ]

        if game_date_matches:
            current_game_date = " ".join(
                game_date_matches
            )

        if game_time_matches:
            current_game_time = " ".join(
                game_time_matches
            )

        if matchup_matches:
            current_matchup = " ".join(
                matchup_matches
            )

        if team_matches:
            current_team = " ".join(
                team_matches
            )

        rows.append(
            {
                "gameDate": current_game_date,
                "gameTime": current_game_time,
                "matchup": current_matchup,
                "team": current_team,
                "playerName": player["text"],
                "status": " ".join(status_matches),
                "reason": " ".join(reason_matches),
            }
        )

    state["gameDate"] = current_game_date
    state["gameTime"] = current_game_time
    state["matchup"] = current_matchup
    state["team"] = current_team

    return rows, state

def extract_report_rows(pdf_path):
    all_rows = []

    state = {
        "gameDate": None,
        "gameTime": None,
        "matchup": None,
        "team": None,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            rows, state = extract_rows_from_page(
                page,
                state=state,
            )

            all_rows.extend(rows)

    return all_rows

def build_injury_report_url(
    report_date,
    report_time,
):
    date_str = report_date.strftime(
        "%Y-%m-%d"
    )

    return (
        f"{NBA_INJURY_REPORT_BASE_URL}/"
        f"Injury-Report_"
        f"{date_str}_"
        f"{report_time}.pdf"
    )

def injury_row_to_availability_update(
    row,
    source_url,
):
    update = AvailabilityUpdate(
        personId=int(row["personId"]),
        team=row["team"],
        status=row["status"],
        reason=row["reason"],
        confidence="high",
        sourceType="nba_official",
        sourceUrl=source_url,
        publishedAt=row["reportPublishedAt"].to_pydatetime(),
    )

    validate_availability_update(update)

    return update

def inspect_report_words(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]

        words = page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        )

        for word in words[:120]:
            print(
                f"text={word['text']!r:25} "
                f"x0={word['x0']:7.1f} "
                f"top={word['top']:7.1f}"
            )

def extract_report_published_at(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]

        words = first_page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        )

    header_words = [
        word["text"]
        for word in words
        if word["top"] < 80
    ]

    header_text = " ".join(header_words)

    marker = "Report:"

    if marker not in header_text:
        raise ValueError(
            "Could not find injury report timestamp."
        )

    timestamp_text = (
        header_text
        .split(marker, 1)[1]
        .strip()
    )

    return datetime.strptime(
        timestamp_text,
        "%m/%d/%y %I:%M %p",
    )

def download_injury_report(
    report_date,
    report_time,
):
    url = build_injury_report_url(
        report_date,
        report_time,
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            ),
            "Referer": "https://official.nba.com/",
        },
        timeout=30,
    )

    response.raise_for_status()

    RAW_INJURY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_INJURY_DIR
        / (
            f"injury_report_"
            f"{report_date:%Y-%m-%d}_"
            f"{report_time}.pdf"
        )
    )

    output_path.write_bytes(
        response.content
    )

    return output_path

def find_latest_injury_report(
    prediction_timestamp,
):
    candidate_time = (
        prediction_timestamp
        .replace(
            second=0,
            microsecond=0,
        )
    )

    # Round down to the nearest 15 minutes.
    minute = (
        candidate_time.minute
        // 15
        * 15
    )

    candidate_time = (
        candidate_time.replace(
            minute=minute
        )
    )

    report_date = (
        prediction_timestamp
    )

    while (
        candidate_time.date()
        == prediction_timestamp.date()
    ):
        report_time = (
            candidate_time.strftime(
                "%I_%M%p"
            )
        )

        url = build_injury_report_url(
            report_date,
            report_time,
        )

        try:
            path = download_injury_report(
                report_date=report_date,
                report_time=report_time,
            )

            print(
                "Using injury report:",
                report_time,
            )

            return (
                path,
                url,
                report_time,
            )

        except requests.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            if status_code not in {
                403,
                404,
            }:
                raise

        candidate_time -= timedelta(
            minutes=15
        )

    raise FileNotFoundError(
        "No official NBA injury report "
        f"found before "
        f"{prediction_timestamp}."
    )

def clean_injury_report_dataframe(df):
    df = df.copy()

    df["status"] = (
        df["status"]
        .str.strip()
        .str.upper()
    )

    df["playerName"] = (
        df["playerName"]
        .str.replace(",", ", ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["team"] = (
        df["team"]
        .str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)
        .str.replace(r"([A-Za-z])(\d)", r"\1 \2", regex=True)
        .str.strip()
    )

    df["reason"] = (
        df["reason"]
        .str.replace("Injury/Illness-", "Injury/Illness - ", regex=False)
        .str.replace("GLeague-", "G League - ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df

def normalize_name(name):
    name = str(name)

    # Remove accents:
    # Dončić -> Doncic
    # Danté -> Dante
    name = unicodedata.normalize(
        "NFKD",
        name,
    )

    name = "".join(
        char
        for char in name
        if not unicodedata.combining(char)
    )

    name = name.lower().strip()

    # Remove suffixes whether separated or attached:
    # "Butler III"   -> "Butler"
    # "McCullarJr."  -> "McCullar"
    # "LivelyII"     -> "Lively"
    # "BagleyIII"    -> "Bagley"
    name = re.sub(
        r"\s*(jr\.?|sr\.?|iii|iv|ii)$",
        "",
        name,
    )

    # Keep only letters and numbers
    name = re.sub(
        r"[^a-z0-9]",
        "",
        name,
    )

    return name

def build_player_lookup(history_df):
    lookup = (
        history_df[
            [
                "personId",
                "firstName",
                "familyName",
            ]
        ]
        .drop_duplicates(
            subset=["personId"]
        )
        .copy()
    )

    lookup["fullName"] = (
        lookup["firstName"].fillna("")
        + " "
        + lookup["familyName"].fillna("")
    )

    lookup["normalizedName"] = (
        lookup["fullName"]
        .apply(normalize_name)
    )

    return lookup

def injury_name_to_normalized(name):
    if "," not in name:
        return normalize_name(name)

    family_name, first_name = name.split(",", 1)

    full_name = (
        first_name.strip()
        + " "
        + family_name.strip()
    )

    return normalize_name(full_name)

def injury_dataframe_to_updates(
    matched_df,
    source_url,
):
    updates = []

    for _, row in matched_df.iterrows():
        if pd.isna(row["personId"]):
            continue

        update = injury_row_to_availability_update(
            row,
            source_url=source_url,
        )

        updates.append(update)

    return updates

def filter_updates_for_game(
    updates,
    team_names,
    prediction_timestamp,
):
    return [
        update
        for update in updates
        if update.team in team_names
        and update.publishedAt <= prediction_timestamp
    ]

def availability_updates_to_dataframe(
    updates,
):
    rows = []

    for update in updates:
        rows.append(
            {
                "personId": update.personId,
                "availabilityStatus": update.status,
                "availabilityReason": update.reason,
                "availabilityConfidence": update.confidence,
                "availabilitySourceType": update.sourceType,
                "availabilitySourceUrl": update.sourceUrl,
                "availabilityPublishedAt": update.publishedAt,
            }
        )

    return pd.DataFrame(rows)

def apply_availability_updates(
    roster_df,
    updates,
):
    roster_df = roster_df.copy()

    updates_df = (
        availability_updates_to_dataframe(
            updates
        )
    )

    if updates_df.empty:
        roster_df["availabilityStatus"] = "UNKNOWN"
        roster_df["availabilityReason"] = None
        roster_df["availabilityConfidence"] = None
        roster_df["availabilitySourceType"] = None
        roster_df["availabilitySourceUrl"] = None
        roster_df["availabilityPublishedAt"] = None

        return roster_df

    roster_df = roster_df.merge(
        updates_df,
        on="personId",
        how="left",
        validate="one_to_one",
    )

    roster_df[
        "availabilityStatus"
    ] = (
        roster_df["availabilityStatus"]
        .fillna("UNKNOWN")
    )

    return roster_df

def retrieve_official_injury_updates(
    prediction_timestamp,
    history_df,
):
    (
        path,
        source_url,
        report_time,
    ) = find_latest_injury_report(
        prediction_timestamp
    )

    df = build_injury_report_dataframe(
        path
    )

    df = clean_injury_report_dataframe(
        df
    )

    history_lookup = build_player_lookup(
        history_df
    )

    master_lookup = (
        build_master_player_lookup()
    )

    df["normalizedName"] = (
        df["playerName"]
        .apply(
            injury_name_to_normalized
        )
    )

    matched = df.merge(
        history_lookup[
            [
                "personId",
                "normalizedName",
            ]
        ],
        on="normalizedName",
        how="left",
        validate="many_to_one",
    )

    matched["matchSource"] = "history"

    matched.loc[
        matched["personId"].isna(),
        "matchSource",
    ] = None

    unmatched_mask = (
        matched["personId"].isna()
    )

    fallback_candidates = (
        master_lookup[
            [
                "personId",
                "normalizedName",
            ]
        ]
        .groupby("normalizedName")
        .agg(
            personId=(
                "personId",
                "first",
            ),
            candidateCount=(
                "personId",
                "nunique",
            ),
        )
        .reset_index()
    )

    fallback = matched.loc[
        unmatched_mask,
        ["normalizedName"],
    ].merge(
        fallback_candidates,
        on="normalizedName",
        how="left",
        validate="many_to_one",
    )

    fallback.loc[
        fallback["candidateCount"] != 1,
        "personId",
    ] = pd.NA

    matched.loc[
        unmatched_mask,
        "personId",
    ] = fallback[
        "personId"
    ].to_numpy()

    matched.loc[
        unmatched_mask
        & matched["personId"].notna(),
        "matchSource",
    ] = "master_directory"

    updates = (
        injury_dataframe_to_updates(
            matched,
            source_url=source_url,
        )
    )

    return updates

if __name__ == "__main__":
    report_date = datetime(
        2025,
        1,
        15,
    )

    path = download_injury_report(
        report_date=report_date,
        report_hour="05PM",
    )

    print("Downloaded:", path)

    # 1. Build + clean injury report dataframe
    df = build_injury_report_dataframe(path)

    df = clean_injury_report_dataframe(df)

    print("Injury report shape:", df.shape)

    # 2. Load historical player data
    history_df = pd.read_csv(
        "data/processed/player_games_2024_25.csv"
    )

    # 3. Build player lookup
    history_lookup = build_player_lookup(
        history_df
    )

    master_lookup = build_master_player_lookup()

    # 4. Normalize injury report names
    df["normalizedName"] = (
        df["playerName"]
        .apply(injury_name_to_normalized)
    )


    # 5. Match to personId
    matched = df.merge(
        history_lookup[
            [
                "personId",
                "normalizedName",
            ]
        ],
        on="normalizedName",
        how="left",
        validate="many_to_one",
    )
    print(
        "Original rows:",
        len(df),
    )

    print(
        "Rows after merge:",
        len(matched),
    )
    matched["matchSource"] = "history"

    matched.loc[
        matched["personId"].isna(),
        "matchSource",
    ] = None

    assert len(matched) == len(df)

    unmatched_mask = matched["personId"].isna()

    fallback_candidates = (
        master_lookup[
            [
                "personId",
                "normalizedName",
            ]
        ]
        .groupby("normalizedName")
        .agg(
            personId=("personId", "first"),
            candidateCount=("personId", "nunique"),
        )
        .reset_index()
    )

    fallback = matched.loc[
        unmatched_mask,
        ["normalizedName"]
    ].merge(
        fallback_candidates,
        on="normalizedName",
        how="left",
        validate="many_to_one",
    )

    fallback.loc[
        fallback["candidateCount"] != 1,
        "personId",
    ] = pd.NA
    fallback_ids = fallback["personId"].to_numpy()

    matched.loc[
        unmatched_mask,
        "personId",
    ] = fallback_ids

    matched.loc[
        unmatched_mask
        & matched["personId"].notna(),
        "matchSource",
    ] = "master_directory"

    matched_count = matched["personId"].notna().sum()

    print()
    print(
        f"Matched: {matched_count}/{len(matched)} "
        f"({matched_count / len(matched):.1%})"
    )
    print()
    print("Match source:")
    print(
        matched["matchSource"]
        .value_counts(dropna=False)
    )

    unmatched = matched[
        matched["personId"].isna()
    ]

    print(
        "Unmatched players:",
        len(unmatched),
    )

    if not unmatched.empty:
        print(
            unmatched[
                [
                    "playerName",
                    "team",
                ]
            ].to_string(index=False)
        )

    source_url = build_injury_report_url(
        report_date=report_date,
        report_hour="05PM",
    )

    first_update = (
        injury_row_to_availability_update(
            matched.iloc[0],
            source_url=source_url,
        )
    )

    print()
    print("First availability update:")
    print(first_update)
    updates = injury_dataframe_to_updates(
        matched,
        source_url=source_url,
    )

    print()
    print("Total availability updates:", len(updates))

    print("\nFirst 5 updates:")
    for update in updates[:5]:
        print(update)

    game_updates = filter_updates_for_game(
        updates,
        team_names={
            "New York Knicks",
            "Philadelphia 76ers",
        },
        prediction_timestamp=datetime(
            2025,
            1,
            15,
            18,
            0,
        ),
    )

    print()
    print("NYK vs PHI updates:")

    for update in game_updates:
        print(update)