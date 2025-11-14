from set_up_scripts import set_up_vot
from models import tuned_models_generation
from models import voting_model


def main():
    print("\n=== Avvio pipeline Pokémon Voting Model ===\n")

    set_up_vot.main()
    tuned_models_generation.main()
    voting_model.main()

    print("\n=== Pipeline completata con successo ===\n")


if __name__ == "__main__":
    main()
