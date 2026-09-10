from nba_api.stats.endpoints import leaguegamelog

def fetch_games():
    game_log = leaguegamelog.LeagueGameLog(
        season="2025-26",
        season_type_all_star="Regular Season",
    )

    df = game_log.get_data_frames()[0]

    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    fetch_games()