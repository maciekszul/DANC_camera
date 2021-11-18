from __future__ import division
import unittest
from math import atan2, degrees

class VisualAngle():
	"""docstring for ClassName"""

	def __init__(self, pix_w, pix_h, cm_w, cm_h, cm_dist):
		self.settings = dict()
		self.settings["pix_h"] = pix_h
		self.settings["pix_w"] = pix_w
		self.settings["cm_h"] = cm_h
		self.settings["cm_w"] = cm_w
		self.settings["cm_dist"] = cm_dist


	def print_all(self):
		"""
		Prints contents of an object line by line
		"""
		k = self.settings.keys()
		for i in k:
			print("{0}: {1}".format(i, self.settings[i]))


	def pixDeg(self):
		"""
		Returns size of a single pixel in degrees
		"""
		pd = degrees(atan2(.5 * self.settings["cm_h"], self.settings["cm_dist"])) / (.5 * self.settings["pix_h"])
		self.settings["pix_in_deg"] = pd
		return pd


	def pixCm(self):
		"""
		Returns size of a pixel in centimeters. 
		"""
		pc = self.settings["cm_h"] / self.settings["pix_h"]
		self.settings["pix_in_cm"] = pc
		return pc


	def cmPix(self):
		"""
		Returns size of a centimeter in pixels. 
		"""
		cp = self.settings["pix_h"] / self.settings["cm_h"]
		self.settings["cm_in_pix"] = cp
		return cp



	def degPix(self):
		"""
		Returns size of a degree in pixels. 
		"""
		dp = 1./ self.pixDeg()
		self.settings["deg_in_pix"] = dp
		return dp


	def degCm(self):
		"""
		Returns size of a degree in centimeters.
		"""
		dc = self.degPix() * self.pixCm()
		self.settings["deg_in_cm"] = dc
		return dc


	def cmDeg(self):
		"""
		Returns size of a centimeter in degrees. 
		"""
		cd = self.cmPix() * self.degPix()
		self.settings["cm_in_deg"] = cd
		return cd


	def stim_deg(self, size, *name):
		"""
		size - in pixels
		optional argument with stimulus name to add to main object
		returns stimulus size in degrees
		if named,
		"""
		s = size * self.pixDeg()
		if name:
			self.settings[name[0]] = s
		return s


	def stim_pix(self, size, *name):
		"""
		size - in degrees
		optional argument with stimulus name to add to main object
		returns stimulus size in pixels, rounded to full pixel.
		"""
		s = round(size * self.degPix())
		if name:
			self.settings[name[0]] = s
		return s


	def dps(self, speed, hz, *name):
		"""
		speed - in degrees per frame of displacement
		returns speed in degrees per second
		optional argument with stimulus name to add to main object
		"""
		s = speed * hz
		if name:
			self.settings[name[0]] = s
		return s

class testVA(unittest.TestCase):
	pass
