import tkinter as tk
import tkinter.ttk
from tkinter import *
import tkinter.messagebox
from tkinter import ttk
from ttkthemes import ThemedTk  # importing all important libraries
import hackathon as H
import numpy as np

topRec = H.getTopRec()

options = [
    'Select Genre',
    'Select Genre',
    'Action',
    'Adventure',
    'Animation',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'History',
    'Horror',
    'Music',
    'Mystery',
    'Romance',
    'Science Fiction',
    'TV Movie',
    'Thriller',
    'War',
    'Western'
]
options2 = ['Select Language', 'Select Language', 'en', 'zh', 'fr', 'it', 'sp', 'hi']
# create root window
root = ThemedTk()

# root window title and dimension4
root.title("The Movie Recommender")

# Set geometry (widthxheight)
root.geometry('1000x400')

# root.iconbitmap('c:/tkinter.com/images/codemy.ico')
style = ttk.Style(root)
our_themes = root.get_themes()


def changer(theme):  # changing themes according to themes
    style.theme_use(theme)


my_menu = Menu(root)
root.config(menu=my_menu)

theme_menu = Menu(my_menu, tearoff=0)
my_menu.add_cascade(label="Themes", menu=theme_menu)  # adding menu

for t in our_themes:
    theme_menu.add_command(label=t, command=lambda t=t: changer(t))  # changing according to a selected themes

# adding a label to the root window
l1 = ttk.Label(root, text="DEFENDER RECOMMENDER SYSTEM")
l1.grid(row=0, column=1)
l2 = ttk.Label(root, text="Select genre of the movie you want to watch : ")
l2.grid(row=1, column=0, sticky=W, pady=2)
l3 = ttk.Label(root, text="Select language of the movie you want to watch : ")
l3.grid(row=1, column=1, sticky=W, pady=2)
# datatype of menu text
clicked = StringVar()
clicked2 = StringVar()
# initial menu text
clicked.set("Select Genre")
clicked2.set("Select Language")
# Create Dropdown menu
drop = ttk.OptionMenu(root, clicked, *options)
drop.grid(row=3, column=0, sticky=W)
drop2 = ttk.OptionMenu(root, clicked2, *options2)
drop2.grid(row=3, column=1, sticky=W)
# recommendation()
# Creating seperators for better UI
x1 = tkinter.ttk.Separator(root, orient=VERTICAL).grid(column=2, row=1, rowspan=12, sticky='ns')

l3 = ttk.Label(root, text="Movie that you would like to watch: ")
l3.grid(row=3, column=3, sticky=W, pady=2)
l11 = Text(height=1, width=15)
l11.grid(row=3, column=4, sticky=W)
# print(topRec.shape[0])
# print(topRec['title'])
# button widget with green color text
l = list()
j = 0
for i in topRec['title']:
    # print(i)
    l.append(ttk.Label(root, text=i))
    l[j].grid(row=2 + j, column=6, sticky=W)
    j += 1


def recommendation():
    genre = clicked.get()
    language = clicked2.get()
    movie = l11.get(1.0, "end-1c")

    if genre != 'Select Genre':
        genre_based = H.create_gen_based(genre)
        isGenre = True
    else:
        isGenre = False
    if language != 'Select Language':
        language_based = H.create_language_based(language)
        isLanguage = True
    else:
        isLanguage = False
    if movie != '':
        movie_based = H.get_recommendations(movie)
        isMovie = True
    else:
        isMovie = False

    rec = list()
    if isLanguage == True and isGenre == False:
        for i in language_based:
            rec.append(i)
    elif isLanguage == False and isGenre == True:
        for i in genre_based:
            rec.append(i)
    elif isLanguage == True and isGenre == True:
        for i in genre_based:
            for j in language_based:
                if i == j:
                    rec.append(i)

    if isMovie == True:
        if isLanguage == False and isGenre == False:
            for i in movie_based:
                rec.append(i)
        else:
            rec2 = list()
            for i in rec:
                for j in movie_based:
                    if i == j:
                        rec2.append(i)
            rec.clear()
            for i in rec2:
                rec.append(i)
    else:
        if isLanguage == False and isGenre == False:
            a = list()
            j = 0
            for i in topRec['title']:
                # print(i)
                a.append(ttk.Label(root, text=i + ' ' * 35))
                a[j].grid(row=2 + j, column=6, sticky=W)
                j += 1
    # print(rec)
    # print(genre_based)
    # print(language_based)
    k = 0
    for i in rec:
        # print(i)
        # l[k].clipboard_clear()
        l[k] = ttk.Label(root, text=i + ' ' * 35)
        l[k].grid(row=2 + k, column=6, sticky=W)
        k += 1
        if k >= 5:
            break
    while k < 5:
        l[k].clipboard_clear()
        l[k] = ttk.Label(root, text=' ' * 75)
        l[k].grid(row=2 + k, column=6, sticky=W)
        k += 1


button = ttk.Button(root, text="SUBMIT", command=recommendation)
button.grid(row=15, column=1, sticky=S, pady=45)

# Creating seperators for better UI
x2 = tkinter.ttk.Separator(root, orient=VERTICAL).grid(column=5, row=1, rowspan=12, sticky='ns')

l16 = ttk.Label(root, text="Results").grid(row=1, column=6, sticky=W, pady=10)
root.mainloop()
