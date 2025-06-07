from google_play_scraper import app,Sort, reviews_all
import csv
import os    


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
                country: str = "et",
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
    

    def export_reviews_to_csv(reviews, csv_filename,folder='data'):
        """
        Export a list of reviews to a CSV file. Appends if the file already exists.

        Args:
            reviews (list): List of review dictionaries (from Google Play).
            csv_filename (str): Path to the CSV file to write or append to.
        """
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        # Full path to the CSV file
        filepath = os.path.join(folder, csv_filename)        
        # Define the fieldnames you want to export
        fieldnames = ['app_id', 'userName', 'score', 'content', 'at']

        # Check if file exists to decide header writing
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only once (if file does not exist)
            if not file_exists:
                writer.writeheader()

            for review in reviews:
                writer.writerow({
                    'app_id': review.get('appId'),
                    'userName': review.get('userName'),
                    'score': review.get('score'),
                    'content': review.get('content'),
                    'at': review.get('at')
                })

        print(f"✅ {len(reviews)} reviews exported to {csv_filename}")


    def export_all_reviews_to_single_csv(self, reviews_list, csv_filename="all_bank_reviews.csv", folder='data'):
            """
            Export reviews from multiple apps to a single CSV file.

            Args:
                reviews_list (list): List of tuples like (app_id, reviews).
                csv_filename (str): Name of the CSV file.
                folder (str): Folder to save the CSV file in.
            """
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, csv_filename)

            fieldnames = ['app_id', 'userName', 'score', 'content', 'at']

            file_exists = os.path.isfile(filepath)
            with open(filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                for app_id, reviews in reviews_list:
                    for r in reviews:
                        writer.writerow({
                            'app_id': app_id,
                            'userName': r.get('userName'),
                            'score': r.get('score'),
                            'content': r.get('content'),
                            'at': r.get('at')
                        })
            print(f"✅ Exported to {filepath}")
