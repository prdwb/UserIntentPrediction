def user_auth(affiliations):
    # Common User; MVP; Community Moderator; Article Author;
    # Microsoft; Forum Owner; Support Engineer
    res = []
    for affiliation in affiliations:
        affi_vector = [0] * 7

        if affiliation.find('Common User') != -1 or affiliation.strip() == '':
            affi_vector[0] = 1
            res.append(list(map(str, affi_vector)))
            continue

        affi_vector[1] = 1 if affiliation.find('MVP') != -1 else 0
        affi_vector[2] = 1 if affiliation.find('Community Moderator') != -1 else 0
        affi_vector[3] = 1 if affiliation.find('Article Author') != -1 else 0
        affi_vector[4] = 1 if affiliation.find('Microsoft') != -1 else 0
        affi_vector[5] = 1 if affiliation.find('Forum Owner') != -1 else 0
        affi_vector[6] = 1 if affiliation.find('Support Engineer') != -1 else 0

        res.append(list(map(str, affi_vector)))
    return res

if __name__ == '__main__':
    print(user_auth(['MVP Community Moderator | Article Author', '']))


