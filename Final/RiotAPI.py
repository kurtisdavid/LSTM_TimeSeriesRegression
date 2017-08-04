import requests
from collections import deque
import simplejson as json
import glob
import time
import os
import sys

# Based code off of this project: https://github.com/pseudonym117/Riot-Watcher/blob/master/riotwatcher/riotwatcher.py
# Used a smaller version, since I only needed certain parts of the API


class LoLException(Exception):
	def __init__(self, error, response):
		self.error = error
		self.headers = response.headers

	def __str__(self):
		return self.error

	def __eq__(self, other):
		if isinstance(other, "".__class__):
			return self.error == other
		elif isinstance(other, self.__class__):
			return self.error == other.error and self.headers == other.headers
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return super(LoLException).__hash__()


error_400 = "Bad request"
error_401 = "Unauthorized"
error_403 = "Blacklisted key"
error_404 = "Game data not found"
error_429 = "Too many requests"
error_500 = "Internal server error"
error_503 = "Service unavailable"
error_504 = 'Gateway timeout'


def raise_status(response):
	if response.status_code == 400:
		raise LoLException(error_400, response)
	elif response.status_code == 401:
		raise LoLException(error_401, response)
	elif response.status_code == 403:
		raise LoLException(error_403, response)
	elif response.status_code == 404:
		raise LoLException(error_404, response)
	elif response.status_code == 429:
		raise LoLException(error_429, response)
	elif response.status_code == 500:
		raise LoLException(error_500, response)
	elif response.status_code == 503:
		raise LoLException(error_503, response)
	elif response.status_code == 504:
		raise LoLException(error_504, response)
	else:
		response.raise_for_status()

def possible(request):

	if request.status_code == 400 or request.status_code == 404 or request.status_code == 415:
		return "skip"
	elif request.status_code == 429:

		print(request.headers)
		type_limit = request.headers.get("X-Rate-Limit-Type", "proxy")
 
		if type_limit == "proxy":
			print("PROXY PROBLEM")
			time.sleep(5)
			return "wait"
		else:
 
			wait_time = request.headers.get("Retry-After", -42)
 
			if wait_time == -42:
 
				print("PROXY PROBLEM")
				time.sleep(5)
				return "wait"
 
			else:
 
				print(type_limit + " HAS EXCEEDED AND MUST WAIT " + str(wait_time) + " SECONDS." )
				time.sleep(int(wait_time))
				return "wait"
		 
	elif request.status_code == 500 or request.status_code == 503:
		return "wait"
	elif request.status_code == 200:
		return "success"
	elif request.status_code == 403:
		return "quit"
	else:
		return "unknown"

# keeps a Deque of times and removes the requests from the front as the time limit decreases
class RateLimiter(object):

	def __init__(self, n_made, n_requests, seconds):

		self.allowed_requests = n_requests
		self.seconds = seconds
		self.made_requests = deque()
		for i in range(n_made):
			self.add_request()

	def __reload(self):
		t = time.time()
		while len(self.made_requests) > 0 and self.made_requests[0] < t:
			self.made_requests.popleft()

	def add_request(self):
		self.made_requests.append(time.time() + self.seconds)

	def request_available(self):

		self.__reload()
		return len(self.made_requests) < self.allowed_requests


class RiotAPI(object):

	def __init__(self, key, limits):

		self.api_key = key
		self.limits = limits

	def can_make_request(self):

		for lim in self.limits:

			if not lim.request_available():

				return False

		return True

	def getMatch(self, matchID):

		while not self.can_make_request():

			time.sleep(1)

		url = "https://na.api.pvp.net/api/lol/na/v2.2/match/" + str(matchID) + "?api_key=" + self.api_key

		request = requests.get(url)

		check = possible(request)
		if check == "skip" or check == "unknown":
			print(str(matchID) + " unsuccessful with error" + str(request.status_code))
			return None
		elif check == "quit":
			print("CARE OF BLACKLIST")
			sys.exit(0)
		elif check == "wait":

			while check!="success":

				while not self.can_make_request():
					time.sleep(1)

				request = requests.get(url)
				check = possible(request)

		for lim in self.limits:

			lim.add_request()

		return request.json()

	def getExtraGames(self, matchJSON, playerID):

		timestamp = matchJSON["matchCreation"]

		game_ids = []

		for participant in matchJSON["participants"]:

			if participant["timeline"]["lane"] == "TOP":

				game_ids.append(participant["participantId"])

		wantedID = 0
		pID = 0
		champID = 0

		for identity in matchJSON["participantIdentities"]:

			if identity["participantId"] in game_ids and str(identity["player"]["summonerId"])!=str(playerID):

				wantedID = identity["player"]["summonerId"]
				pID = identity["participantId"]
				break

		for participant in matchJSON["participants"]:

			if participant["participantId"] == pID:

				champID = participant["championId"]
				break

		return self.getSpecificMatchList(wantedID, champID, timestamp-1)

		

	def getSpecificMatchList(self, summonerID, champID, timestamp):

		while not self.can_make_request():

			time.sleep(1)

		url = "https://na.api.pvp.net/api/lol/na/v2.2/matchlist/by-summoner/" + str(summonerID) + "?championIds=" + str(champID) + "&rankedQueues=TEAM_BUILDER_DRAFT_RANKED_5x5,RANKED_TEAM_5x5&seasons=SEASON2016&endTime="  + str(timestamp) + "&api_key=" + str(self.api_key)

		request = requests.get(url)

		check = possible(request)
		if check == "skip" or check == "unknown":
			print(str(summonerID) + " unsuccessful with error" + str(request.status_code))
			return None
		elif check == "quit":
			print("CARE OF BLACKLIST")
			sys.exit(0)
		elif check == "wait":

			while check!="success":

				while not self.can_make_request():
					time.sleep(1)

				request = requests.get(url)
				check = possible(request)


		for lim in self.limits:

			lim.add_request()

		return request.json()


	def getMatchList(self, summonerID):

		while not self.can_make_request():

			time.sleep(1)

		url = "https://na.api.pvp.net/api/lol/na/v2.2/matchlist/by-summoner/" + str(summonerID) + "?api_key=" + self.api_key

		request = requests.get(url)

		check = possible(request)
		if check == "skip" or check == "unknown":
			print(str(summonerID) + " unsuccessful with error" + str(request.status_code))
			return None
		elif check == "quit":
			print("CARE OF BLACKLIST")
			sys.exit(0)
		elif check == "wait":

			while check!="success":

				while not self.can_make_request():
					time.sleep(1)

				request = requests.get(url)
				check = possible(request)

		for lim in self.limits:

			lim.add_request()

		return request.json()


def getMatches(API):

	players = [66271229, 44207669, 47063254, 56723622, 34399881, 47837396, 19305039, 37348109, 51635691, 28360391]

	for i in range(8, len(players)):

		directory = "./" + str(players[i])

		if not os.path.exists(directory):
			os.makedirs(directory)

		playerID = players[i]

		matchlist = API.getMatchList(playerID)

		for match in matchlist["matches"]:

			poss_queues = ["TEAM_BUILDER_DRAFT_RANKED_5x5", "RANKED_TEAM_5x5"]
			season = "SEASON2016"
			lane = "TOP"

			matchId = match["matchId"]

			

			if match["queue"] in poss_queues and match["season"] == season and match["lane"] == lane:

				realMatch = API.getMatch(matchId)
				print(matchId)

				with open("./" + str(playerID) + "/" + str(matchId) + ".json", "w") as f:

					json.dump(realMatch,f)

def getExtras(API):

	players = [66271229, 44207669, 47063254, 56723622, 34399881, 47837396, 19305039, 37348109, 51635691, 28360391]

	for i in range(0,1):

		player = players[i]

		DATA_FILES = glob.glob('./' + str(player) + '/*.json')

		for filename in DATA_FILES:

			new_folder = filename.split("\\")[1].split(".")[0]
			print("~~~~~~~~~~~~~~~")
			print(str(player) + " - " + new_folder)



			games = None

			with open(filename) as data_file:

				data = json.load(data_file)

				games = API.getExtraGames(data, player)

			if games is None or games.get("matches") is None:

				continue

			count = 0
			if not os.path.exists("./" + str(player) + "/" + str(new_folder)):
				os.makedirs("./" + str(player) + "/" + str(new_folder))
			for match in games["matches"]:

				if count>=24:

					break

				if match["lane"] == "TOP":

					count += 1
					matchId = match["matchId"]
					print(matchId)
					realMatch = API.getMatch(matchId)

					with open("./" + str(player) + "/" + str(new_folder) +  "/" + str(matchId) + ".json", "w") as f:

						json.dump(realMatch,f)

def fixNulls(API):

	null_file = open("nulls.txt")

	for line in null_file:

		directory = line[:-1]

		if os.path.exists(directory):
			os.remove(directory)

		print(directory)

		matchId = directory.split("/")[-1].split(".")[0]

		json_data = API.getMatch(matchId)

		with open(directory, "w") as f:

			json.dump(json_data,f)



if __name__ == "__main__":

	config = open("config.txt")

	api_key = config.readline()[:-1]
	print(api_key)

	API = RiotAPI(api_key, limits = (RateLimiter(1,10,10), RateLimiter(10,500,600)))

	# getMatches(API)
	fixNulls(API)

	# myID = 39550290

	# matchIds = open("matchIDs.txt")

	# for ID in matchIds:

	# 	ID = ID[:-1]

	# 	if ID != "":

	# 		match = API.getMatch(ID)

	# 		with open("./Sensen/" + ID + ".json", "w") as f:

	# 			json.dump(match,f)

	# DATA_FILES = glob.glob('./Sensen/*.json')

	# for filename in DATA_FILES:

	# 	print(filename)

	# 	file = open(filename)
	# 	json_data = json.load(file)
	# 	file.close()

	# 	if filename == "./Sensen/2079308613.json":

	# 		print(json_data)

	# 	participantIdentities = json_data["participantIdentities"]

	# 	player_IDs = {}

	# 	for participant in participantIdentities:

	# 		player_IDs[participant["participantId"]] = participant["player"]["summonerId"]


	# 	participantData = json_data["participants"]

	# 	desiredId = 0

	# 	for participant in participantData:

	# 		timeline = participant["timeline"]
	# 		lane = timeline["lane"]

	# 		if lane == "TOP" and player_IDs[participant["participantId"]]!=myID:

	# 			desiredId = player_IDs[participant["participantId"]]
	# 			break


	# 	if desiredId == 0:

	# 		continue

	# 	timestamp = json_data["matchCreation"]

	# 	matchList = API.getMatchList(desiredId)

	# 	matchList = matchList["matches"]

	# 	for match in matchList:

	# 		matchTime = match["timestamp"]
	# 		queue = "TEAM_BUILDER_DRAFT_RANKED_5x5"
	# 		season = "SEASON2016"
	# 		lane = "TOP"

	# 		matchId = match["matchId"]

	# 		if matchTime<timestamp and match["queue"] == queue and match["season"] == season and match["lane"] == lane:

	# 			realMatch = API.getMatch(matchId)

	# 			with open("./Sensen/" + str(desiredId) + "/" + str(matchId) + ".json", "w") as f:

	# 				json.dump(realMatch,f)






