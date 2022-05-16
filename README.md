# Speech-Emotion-Recognition
Expert System for Determining the Dominant Expressed Emotion from the Voice - Bachelor's thesis 

## Spuštění demonstrační aplikace
Demonstrační aplikaci je možné spustit po nainstalování potřebných knihoven přímo ze složky, kde se nachází.

Knihovny, které v projektu používám lze nainstalovat příkazem `pip install -r ./requirements.txt`. Pro tenzorové operace je možné použít Nvidia GPU, což zrychlí dobu trénování sítě.
GPU však musí podporovat CUDA Toolkit. Pokud grafická karta CUDU podporuje, je nutné používat ovladače a software kompatibilní s knihovnou TensorFlow. Dokumentace TensorFlow uvádí [tabulku kompatibilních verzí](https://www.tensorflow.org/install/source#gpu).
Dále Nvidia uvádí [tabulku kompatibilních ovladačů grafických karet s danou verzí CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).

Pro trénování projektu jsem použil grafickou kartu Nvidia GeForce MX130. Nejprve jsem aktivoval ovladač nvidia-driver-470. Následně jsem stáhnul a nainstaloval CUDA 11.2 a poté cuDNN 8.1 (pro stažení je nutné vytvořit Nvidia Developer účet).

## Trénování, validace a testování sítě
Jelikož v projektu nejsou obsaženy vygenerované příznaky, které jsou nezbytné pro trénování sítě (kvůli jejich množství a velikosti), je nutné tyto příznaky nejprve vygenerovat spuštěním *features_generator.py*, případně pomocí *test_features_generator.py*, *test_features_generator_valid.py* a *test_features_generator_noise.py*.
Teprve po vygenerování souborů je možné zahájit trénování sítě spuštěním *train.py*. Případně testování spuštěním *test_train.py*.

## Popis struktury projektu

- *demo.py*
  - Skript obsahuje demonstrační aplikaci. Po spuštění ze složky, kde se nachází, načte příslušné knihovny a požádá uživatele o volbu natrénovaného modelu (které se nachází v podsložkách src1 nebo src2). Následně ve smyčce nahrává zvuk z mikrofonu, vytváří predikce podle načteného modelu a vypisuje je na standardní výstup.
  - Vzhledem k velkému množství varování (v mém případě týkajících se GPU), které produkuje Keras, jsem zakázal jejich vypisování.

- *datasets*
  - Tato složka obsahuje podsložky s nahrávkami jednotlivých datasetů. Nahrávky jsou už převedené do formátu wav, jsou podvzorkované a jsou z nich vystřihnuté neznělé úseky, které jsou obsaženy na začátku a na konci v původních nahrávkách.
  - Dále jsou ve složce obsaženy 3 shellové skripty, které konvertují původní nahrávky. Jestliže by v podsložkách byly původní soubory datasetů, pak skript připraví nahrávky do takové podoby, v jaké jsou nyní uložené.
  - Není žádoucí tyto skripty nyní spouštět a jakkoliv obsah měnit.

- *src1*
  - Složka obsahuje kódy pro první část bakalářské práce, která je popsána v kapitole 6 "Síť s kombinací několika druhů příznaků".
  - Složka *model* obsahuje mnou natrénovaný model, stejně tak složka model_test obsahuje model, který vznikl při testování sítě.
  - Složka *images* obsahuje vygenerované obrázky, které vznikly pro vizualizaci jednotlivých extrahovaných příznaků nebo při trénování sítě. Tyto obrázky jsou rovněž použity v bakalářské práci.
  - Složky *features*, *test*, *test_valid* a *test_noise* jsou prázdné. Slouží pro uložení vygenerovaných tenzorů, které následně vstupují do neuronové sítě. Proto před trénováním sítě je nutné nejdříve tenzory vygenerovat.
  - Skript *augmentation.py* obsahuje funkce pro augmentaci, *augmentation_play.py* slouží pro přehrání augmentované nahrávky danou funkcí z augmentation.py.
  - Skript *datasets.py* obsahuje slovníky pro dekódování emoce z názvu soboru každého datasetu. Zároveň obsahuje funkce, které načtou příslušné soubory uložené ve složce datasets.
  - Skript *features.py* obsahuje funkci, která ze zvukové vlny sestaví a vrátí vektor příznaků.
  - Pro generování příznaků (souborů ve složce *src1/features*) je nutné spustit skript *features_generator.py*. Tento skript převede wav soubory do souborů obsahující vstupní vektory neuronové sítě. Aby bylo rozeznatelné, do jaké třídy vektory patří, uloží se soubory do složek s názvem příslušné třídy.
  - Po vygenerování těchto souborů je možné spustit skript *train.py*, který slouží pro trénování a validaci sítě.
  - Podobně pro testování sítě je nutné vygenerovat vstupní vektory pomocí generátorů *test_features_generator.py*, *test_features_generator_valid.py* a *test_features_generator_noise.py*. A poté je možné spustit skript *test_train.py*.
  - Nakonec jsou ve složce *src1* obsaženy soubory, jejichž název začíná *plot_\**. Skripty jsou samostatně spustitelné a slouží pro vizualizaci rozdělení datasetu nebo použitých extrahovaných příznaků.

- *src2*
  - Tato složka obsahuje kódy pro druhou část bakalářské práce popsanou v kapitole 7 "Síť trénovaná na spektrogramu".
  - Struktura složky je stejná jako u *src1*, pouze se liší obsah souborů (*features.py*, *train.py*, *test_train.py*).
