from google_play_scraper import app,Sort, reviews_all
    


class DataCollectionPreProcesse:

    def app_detail(self, id: str, lang: str = 'en', country: str = 'ET'):
        """
        Retrieve application details for a given app ID, language, and country.

        Args:
            id (str): The unique identifier of the application (non-empty).
            lang (str): 2-letter language code (default is 'en').
            country (str): 2-letter country code (default is 'us').

        Returns:
            dict or object: The application details corresponding to the given ID.

        Raises:
            ValueError: If ID is empty, or lang/country codes are invalid.
            TypeError: If any of the inputs are not strings.
        """
        print(id)
        print(lang)
        print(country)
        # Validate `id`
        if not isinstance(id, str):
            raise TypeError("ID must be a string.")
        if not id.strip():
            raise ValueError("ID string cannot be empty.")

        # Validate `lang`
        if not isinstance(lang, str):
            raise TypeError("Language code must be a string.")
        if len(lang) != 2 or not lang.isalpha():
            raise ValueError("Language code must be a 2-letter alphabetic string.")

        # Validate `country`
        if not isinstance(country, str):
            raise TypeError("Country code must be a string.")


        # Call the external function
        return app(id, lang.lower(), country.upper())



    def reviews(self, id: str,
                sleep_milliseconds: int = 0,
                lang: str = 'en',
                country: str = "us",
                sort = Sort.MOST_RELEVANT,
                filter_score_with: int = None):
        """
        Fetches all reviews for a given app from Google Play.

        Args:
            id (str): The app package name (e.g., 'com.dashen.dashensuperapp').
            sleep_milliseconds (int): Time to wait between requests.
            lang (str): Language code, like 'en'.
            country (str): Country code, like 'us'.
            sort (Sort): Sorting strategy (e.g., Sort.NEWEST).
            filter_score_with (int): Optional star rating filter (1 to 5).

        Returns:
            list: A list of review dictionaries.
        """
        return reviews_all(
            id,
            sleep_milliseconds=sleep_milliseconds,
            lang=lang.lower(),
            country=country.upper(),
            sort=sort,
            filter_score_with=filter_score_with
        )


