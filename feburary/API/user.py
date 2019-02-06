from werkzeug.security import generate_password_hash, check_password_hash
import json 

with open('data/users.json', 'r') as f:
    users_dict = json.load(f)


def authenticate(user, password):

    user_list = users_dict['user_list']
    if user in not user_list:
        print('Not in DB')
        return none 
    else:
        pass_db = users_dict['users'][user]
        if check_password_hash(pass_db, password):
            print('In db')
            return True

authenticate('1','12')
# print((authenticate('Ben', 1234))