from set_up_scripts import set_up
from models import tuned_models_generation
from models import stacking_model_generation
from models import logistic


def main():
    print("\n=== Avvio pipeline Pokémon Battles ===\n")

    set_up.main()
    tuned_models_generation.main()
    stacking_model_generation.main()
    logistic.main()

    print("\n=== Pipeline completata con successo ===\n")


if __name__ == "__main__":
    main()
