# This dataset is collected from Yelp.com and first used by Mukherjee et al. 
# This data includes 67,395 reviews for a set of hotels and restaurants in the Chicago area. 
# Reviews include product and user information, timestamp, ratings, and a plaintext review. 
# This dataset contains reviews from 201 hotels and restaurants by 38,063 reviewers. 
# Yelp has a filtering algorithm in place that identifies fake/suspicious reviews and separates them into a filtered list. 
# The filtered reviews are also made public; 
# the Yelp page of a business shows the recommended reviews, while it is also possible to view the filtered/unrecommended reviews through a link at the bottom of the page. 
# While the Yelp anti-fraud filter is not perfect (hence the “near” ground truth), it has been found to produce accurate results 
# (K. Weise. A Lie Detector Test for Online Reviewers, 2011. https://bloom.bg/1KAxzhK.). This Yelp dataset contains both recommended and filtered reviews. We consider them as genuine and fake, respectively. We also separate the users into two classes; spammers: authors of fake (filtered) reviews, and benign: authors with no filtered reviews.
