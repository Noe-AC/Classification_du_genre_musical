import glob
from pydub import AudioSegment
import numpy as np
from scipy.fftpack import fft # transfo de Fourier
from scipy import signal


# Les artistes consirérés
# Directory des artistes
path_artists = "/Users/user_name/Music/iTunes/iTunes Media/Music/"
# Classique (369 morceaux, 8 artistes)
artistes_classique = ["Beethoven","Chopin","Mozart","Satie","Schubert","Shostakovich","Strauß","Vivaldi"]
# Électronique (364 morceaux, 16 artistes)
artistes_electronique = ["Crystal Castles","Daft Punk","Flume","Flying Lotus","Grimes","J.M. Jarre","J.P. Decerf","Justice","The Knife","Kraftwerk","Mort Garson","Mr. Oizo","Police des Moeurs","Technotronic","Trentemøller","Trust"]
# Métal (358 morceaux, 13 artistes)
artistes_metal = ["Accept","AC_DC","All That Remains","Black Sabbath","Cacophony","Iron Maiden","Megadeth","Metallica","Peste Noire","Sepultura","Slipknot","Vengeful","Venom"]
# Ça fait 369+362+358=1091 morceaux

# Bonus : 47 morceaux de vaporwave
artistes_vaporwave = ["Chuck Person","MACINTOSH PLUS","SAINT PEPSI","仮想___ Vを（HS高校__"]

# Pour extraire classique, électronique et métal, décommenter ceci :
artistes = {0:artistes_classique,1:artistes_electronique,2:artistes_metal}
# Pour extraire vaporwave, décommencer ceci :
#artistes = {3:artistes_vaporwave}


# Les genres :
genres = list(artistes.keys())


"""
En fait on enlève :
- 21 morceaux de Chopin, Noctures, car c'est du 88200 et non du 44100
- 1 morceaux électro car max du spectre est nul.

Voilà, selon le compteur j'ai juste 1069 morceaux.
Ce qui concorde avec 1091-22=1069.
Bref :
- Classique : 369-21 = 348 morceaux
- Électronique : 363 morceaux
- Métal : 358 morceaux
Total : 1069 morceaux
"""


# On fait une boucle sur les morceaux
nombre_de_morceaux = 0
skipped_songs = []
skipped_albums = set()
genre = 0
id_number = 0
write_file = open("data.csv","w+") # fichier d'écriture

# On met le header dans le fichier d'écriture
N = 1024 # Number of point in the fft
freq_min = 0
freq_max = 20000
FR = 44100 / N # résolution de la fréquence, e.g. FR = 44100/512 = 86.13 Hz de résolution, 44100/1024=43Hz
longueur_freq = int((freq_max - freq_min) / FR) # nombre de points de fréquence qu'on aura. (20000-0)/(44100/512) = 232.2, (20000-0)/(44100/1024) = 464.4
numbers = list(range(1,longueur_freq+1))
strings = ["%d" % number for number in numbers]
line = "id,genre," + ",".join(strings) + "\n"
write_file.write(line)

seconde_debut = 12
nombre_de_secondes = 1
seconde_fin   = seconde_debut + nombre_de_secondes


# On reprend à :
#id_number = 59


# La grande boucle
for genre in genres:
	for artist in artistes[genre]:
		path_artist = path_artists + artist + "/*"
		albums = glob.glob(path_artist)
		albums.sort()
		print("\nNom de l'artiste : ",artist)
		print("Nombre d'albums : ",len(albums))
		for album in albums:
			morceaux = glob.glob(album + "/*")
			morceaux.sort()
			print("\n  Nom de l'album : ",album[len(path_artist)-1:],"\tNombre de morceaux : ",len(morceaux))
			print("  Nombre de morceaux : ",len(morceaux))
			nombre_de_morceaux += len(morceaux)
			for morceau in morceaux:
				#print("    ",morceau)
				song = AudioSegment.from_file(morceau)
				song = song[seconde_debut*1000:seconde_fin*1000] # on ne garde que de la 4ième à la 5ième seconde
				Audiodata = song.get_array_of_samples()
				Audio_mono = []
				for i in range(int(len(Audiodata)/2)): # on rend ça mono
				    average = (Audiodata[2*i]+Audiodata[2*i+1])/2
				    Audio_mono.append(average)
				n  = len(Audio_mono) # nombre de points
				fs = n / nombre_de_secondes # fréquence d'échantillonage, doit être 44100
				if fs!= 44100:
					print("    ",morceau[len(album)+1:])
					print("Erreur. La fréq. d'échantillonage doit être 44100 mais elle est de :",fs)
					skipped_songs.append(morceau[len(album)+1:])
					skipped_albums.add(album)
				if fs==44100:
					# On manipule le signal :
					f, t, spectrum = signal.spectrogram(np.array(Audio_mono), fs,window = signal.blackman(N),nfft=N)
					spectrum_scaled = 10*np.log10(spectrum)
					average_spectrum = spectrum_scaled.mean(axis=1) # On fait la moyenne du spectre dans le temps
					average_spectrum_cut = average_spectrum[:longueur_freq] # on ne garde que longueur_freq points
					maximum = np.max(average_spectrum_cut)
					if maximum==-np.inf:
						print("    ","\tmax =",maximum,"\t",morceau[len(album)+1:])
						print("Erreur. max spectral nul.")
						skipped_songs.append(morceau[len(album)+1:])
						skipped_albums.add(album)
					else:
						id_number += 1
						print("    id =",id_number,"\tmaximum =",maximum,"\t",morceau[len(album)+1:])
						average_spectrum_normalized = average_spectrum_cut/maximum # on normalise
						maximum = np.max(average_spectrum_normalized) # devrait être égal à 1
						minimum = np.min(average_spectrum_normalized)
						gap = maximum-minimum
						ad_hoc_coefficient = 0.7
						ajustement = np.linspace(0,ad_hoc_coefficient*gap,len(average_spectrum_normalized))
						average_spectrum_ajuste = average_spectrum_normalized+ajustement
						# Maintenant on imprime ça dans un fichier CSV
						# On met les fréquences dans le fichier d'écriture
						#strings = ["%.6f" % number for number in average_spectrum_normalized]
						strings = ["%.6f" % number for number in average_spectrum_ajuste]
						line = str(id_number) + "," + str(genre) + "," + ",".join(strings) + "\n"
						write_file.write(line)

# On ferme le fichier d'écriture
write_file.close() 

print("Nombre de morceaux gardés : ",nombre_de_morceaux)
print("Nombre de morceaux skipped : ",len(skipped_songs))
print("Morceaux skipped : ",skipped_songs)
print("Albums skipped :",skipped_albums)

"""
Nombre de morceaux gardés :  1091
moins ce qui est skipped
"""

"""
À faire :
- gérer les plus grosses fréquences d'échantillonage que 44100
"""


"""
Nombre de morceaux gardés :  1091
Nombre de morceaux skipped :  22
Morceaux skipped :
['01 Nocturne op.48 à Mademoiselle Laude Duperré N°1.m4a',
'02 Nocturne op.48 à Mademoiselle Laude Duperré N°2.m4a',
'03 Nocturne op.15 à Monsieur F. Hiller N°1.m4a',
'04 Nocturne op.15 à Monsieur F. Hiller N°2.m4a',
'05 Nocturne op.15 à Monsieur F. Hiller N°3.m4a',
"06 Nocturne op.27 à la comtesse d'Appony N°1.m4a",
"07 Nocturne op.27 à la comtesse d'Appony N°2.m4a",
'08 Nocturne op.20 en ut dièse mineur.m4a',
'09 Nocturne op.32 à la baronne de Billing N°1.m4a',
'10 Nocturne op.32 à la baronne de Billing N°2.m4a',
'11 Nocturne op.55 à Mademoiselle J.W.Stirling N°1.m4a',
'12 Nocturne op.55 à Mademoiselle J.W.Stirling N°2.m4a',
'13 Nocturne op.37 N°1.m4a', '14 Nocturne op.37 N°2.m4a',
'15 Nocturne op.9 à Madame Camille Pleyel N°1.m4a',
'16 Nocturne op.9 à Madame Camille Pleyel N°2.m4a',
'17 Nocturne op.9 à Madame Camille Pleyel N°3.m4a',
'18 Nocturne op. posthume N°19 op.72 n°1 en mi mineur.m4a',
'19 Nocturne op. posthume en do mineur.m4a',
'20 Nocturne op.62 à Mademoiselle R. de Könneritz N°1.m4a',
'21 Nocturne op.62 à Mademoiselle R. de Könneritz N°2.m4a',
'1-02 Vamp.mp3']
Albums skipped : {'/Users/nac/Music/iTunes/iTunes Media/Music/Trentemøller/The Last Resort',
'/Users/nac/Music/iTunes/iTunes Media/Music/Chopin/Chopin - Nocturnes (François Chaplin)'}

il reste 1069 morceaux
"""












