from nba_api.stats.static import players


def main():
    all_players = players.get_players()

    print("Total players:", len(all_players))

    print("\nFirst 10:")
    for player in all_players[:10]:
        print(player)

    targets = [
        "DaRon Holmes II",
        "Saddiq Bey",
        "Bojan Bogdanovic",
    ]

    print("\nTarget searches:")

    for target in targets:
        matches = [
            player
            for player in all_players
            if target.lower()
            in player["full_name"].lower()
        ]

        print(f"\n{target}:")
        for match in matches:
            print(match)


if __name__ == "__main__":
    main()