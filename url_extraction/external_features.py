# external_features.py

from datetime import datetime
import whois
import time
import re

#################################################################################################################################
#               Check if the domain is recognized by WHOIS
#################################################################################################################################

def whois_registered_domain(domain):
    """
    Checks if the given domain matches its WHOIS registered domain name.

    Parameters:
        domain (str): The domain name to check.

    Returns:
        int: 0 if the WHOIS registered domain matches the given domain, otherwise 1.
             Returns 1 if WHOIS lookup fails.
    """
    try:
        hostname = whois.whois(domain).domain_name
        if type(hostname) == list:
            for host in hostname:
                if re.search(host.lower(), domain):
                    return 0
            return 1
        else:
            if re.search(hostname.lower(), domain):
                return 0
            else:
                return 1     
    except:
        return 1

#################################################################################################################################
#               Check and returns the domain's registration length/age
#################################################################################################################################

def domain_registration_length(domain):
    """
    Returns the number of days until the domain's expiration date.

    Parameters:
        domain (str): The domain name to check.

    Returns:
        int: The number of days until expiration if available, 0 if no expiration date is found, 
             or -1 if WHOIS lookup fails.
    """
    try:
        res = whois.whois(domain)
        expiration_date = res.expiration_date
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        # Some domains do not have expiration dates. The application should not raise an error if this is the case.
        if expiration_date:
            if type(expiration_date) == list:
                expiration_date = min(expiration_date)
            return abs((expiration_date - today).days)
        else:
            return 0
    except:
        return -1

#################################################################################################################################
#               Check and returns the domain's age of a url
#################################################################################################################################

def domain_age(domain):
    """
    Returns the age of the domain in days.

    Parameters:
        domain (str): The domain name to check.

    Returns:
        int: The number of days since the domain's creation date, 
             -2 if the creation date is unavailable, 
             or -1 if WHOIS lookup fails.
    """
    try:
        url = domain.split("//")[-1].split("/")[0].split('?')[0]
        domain_info = whois.whois(url)

        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            age_in_days = (datetime.utcnow() - creation_date).days
            return age_in_days
        else:
            return -2
    except:
        return -1