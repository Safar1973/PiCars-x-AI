#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computer_Vision.py - Modul für die Bildverarbeitung und Torerkennung des PiCar-X
"""

from picarx import Picarx
import cv2
import numpy as np
import time
from vilib import Vilib

class TorErkennung:
    def __init__(self, px):
        """
        Initialisiert die Torerkennung
        
        Args:
            px: Picarx-Objekt
        """
        self.px = px
        self.camera_width = 320
        self.camera_height = 240
        
        # Parameter für die Torerkennung
        self.tor_breite_mm = 210  # Breite des Tors in mm
        self.parkabstand_mm = 100  # Parkabstand vor dem Tor in mm (10 cm)
        self.tor_farbe_hsv_lower = np.array([0, 0, 0])  # Schwarz/Dunkel
        self.tor_farbe_hsv_upper = np.array([180, 255, 50])
        
        # Kamera-Parameter
        self.focal_length = 290  # Geschätzter Brennweitenwert für die Kamera
        
        # Kamera initialisieren
        self._setup_camera()
        
        # Status-Variablen
        self.tor_erkannt = False
        self.tor_position = 0  # Position des Tors (-1 bis 1, 0 ist Mitte)
        self.tor_entfernung = 0  # Geschätzte Entfernung zum Tor in mm
        self.tor_winkel = 0  # Geschätzter Winkel zum Tor in Grad
        
        # Filter für Entfernungsmessung
        self.distance_history = []
        self.max_distance_history = 5
        
        # Kalibrierungsfaktor für kleine Objekte
        self.small_object_calibration = 1.2  # Faktor zur Korrektur der Entfernungsschätzung bei kleinen Objekten
    
    def _setup_camera(self):
        """Kamera initialisieren und Bildverarbeitung starten"""
        print("Kamera wird initialisiert...")
        try:
            Vilib.camera_start(vflip=False, hflip=False)
            Vilib.display(local=True, web=False)
            time.sleep(2)  # Warten auf Kamera-Initialisierung
            print("Kamera bereit!")
        except Exception as e:
            print(f"Fehler bei der Kamera-Initialisierung: {e}")
            raise
    
    def _filter_distance(self, distance):
        """
        Filtert die gemessene Entfernung für stabilere Werte
        
        Args:
            distance: Gemessene Entfernung in mm
            
        Returns:
            float: Gefilterte Entfernung in mm
        """
        # Füge neue Messung zur Historie hinzu
        self.distance_history.append(distance)
        
        # Begrenze die Größe der Historie
        if len(self.distance_history) > self.max_distance_history:
            self.distance_history.pop(0)
        
        # Median-Filter für robustere Entfernungsmessung
        return np.median(self.distance_history)
    
    def detect_gate(self):
        """
        Erkennt das Tor im Kamerabild
        
        Returns:
            bool: True wenn Tor erkannt wurde, sonst False
            float: Position des Tors (-1 bis 1, 0 ist Mitte)
            float: Geschätzte Entfernung zum Tor in mm
            float: Geschätzter Winkel zum Tor in Grad
        """
        try:
            # Bild von Vilib holen
            if hasattr(Vilib, 'img') and Vilib.img is not None:
                frame = Vilib.img.copy()
                
                # In HSV-Farbraum konvertieren
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Maske für dunkle Farbe erstellen (Tor-Pfosten)
                mask = cv2.inRange(hsv, self.tor_farbe_hsv_lower, self.tor_farbe_hsv_upper)
                
                # Rauschen entfernen
                kernel = np.ones((3, 3), np.uint8)  # Kleinerer Kernel für präzisere Erkennung
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Konturen finden
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Konturen nach Größe filtern (für kleine Objekte angepasst)
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Niedrigerer Schwellenwert
                
                if len(valid_contours) >= 2:
                    # Sortieren nach x-Position
                    sorted_contours = sorted(valid_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
                    
                    # Die beiden äußersten Konturen nehmen (links und rechts)
                    left_contour = sorted_contours[0]
                    right_contour = sorted_contours[-1]
                    
                    # Bounding Boxes berechnen
                    x1, y1, w1, h1 = cv2.boundingRect(left_contour)
                    x2, y2, w2, h2 = cv2.boundingRect(right_contour)
                    
                    # Mittelpunkt des Tors berechnen
                    gate_center_x = (x1 + w1/2 + x2 + w2/2) / 2
                    
                    # Breite des Tors in Pixeln
                    gate_width_px = (x2 + w2) - x1
                    
                    # Position relativ zur Bildmitte (-1 bis 1)
                    gate_position = (gate_center_x - self.camera_width/2) / (self.camera_width/2)
                    
                    # Entfernung zum Tor schätzen (mit Brennweite und bekannter Torbreite)
                    # Formel: Entfernung = (Objektbreite_real * Brennweite) / Objektbreite_pixel
                    # Mit Kalibrierungsfaktor für kleine Objekte
                    raw_distance = (self.tor_breite_mm * self.focal_length) / (gate_width_px * self.small_object_calibration)
                    
                    # Entfernung filtern für stabilere Werte
                    distance = self._filter_distance(raw_distance)
                    
                    # Winkel zum Tor berechnen
                    angle = np.arctan(gate_position * self.camera_width / (2 * self.focal_length)) * 180 / np.pi
                    
                    self.tor_erkannt = True
                    self.tor_position = gate_position
                    self.tor_entfernung = distance
                    self.tor_winkel = angle
                    
                    print(f"Tor erkannt: Position = {gate_position:.2f}, Entfernung = {distance:.0f}mm, Winkel = {angle:.1f}°")
                    return True, gate_position, distance, angle
                else:
                    self.tor_erkannt = False
                    print("Kein Tor erkannt")
                    return False, 0, 0, 0
            else:
                print("Kein Bild verfügbar")
                self.tor_erkannt = False
                return False, 0, 0, 0
        except Exception as e:
            print(f"Fehler bei der Torerkennung: {e}")
            self.tor_erkannt = False
            return False, 0, 0, 0
    
    def align_with_gate(self, speed=15):
        """
        Richtet das Fahrzeug zum Tor aus
        
        Args:
            speed: Drehgeschwindigkeit
        
        Returns:
            bool: True wenn Ausrichtung erfolgreich, sonst False
        """
        max_attempts = 20
        attempts = 0
        
        while attempts < max_attempts:
            # Tor erkennen
            gate_found, position, _, angle = self.detect_gate()
            
            if gate_found:
                # Wenn das Tor fast mittig ist, sind wir ausgerichtet
                if abs(position) < 0.05:  # Präzisere Ausrichtung für schmales Tor
                    self.px.stop()
                    print("Fahrzeug ist zum Tor ausgerichtet")
                    return True
                
                # Drehen in Richtung des Tors
                if position < 0:
                    # Tor ist links
                    steering_angle = max(-35, -20 * abs(position))  # Proportionale Steuerung
                    self.px.set_dir_servo_angle(steering_angle)
                    self.px.forward(speed * min(1, abs(position) + 0.3))
                else:
                    # Tor ist rechts
                    steering_angle = min(35, 20 * abs(position))  # Proportionale Steuerung
                    self.px.set_dir_servo_angle(steering_angle)
                    self.px.forward(speed * min(1, abs(position) + 0.3))
                
                time.sleep(0.1)
                self.px.stop()
            else:
                # Wenn kein Tor erkannt, systematisch suchen
                search_angles = [0, 10, -10, 20, -20, 30, -30]
                current_angle = search_angles[attempts % len(search_angles)]
                
                self.px.set_dir_servo_angle(current_angle)
                self.px.forward(speed * 0.7)
                time.sleep(0.2)
                self.px.stop()
            
            attempts += 1
        
        self.px.stop()
        print("Konnte das Tor nicht ausrichten")
        return False
    
    def is_at_parking_distance(self):
        """
        Prüft, ob das Fahrzeug im gewünschten Parkabstand zum Tor ist
        
        Returns:
            bool: True wenn das Fahrzeug im Parkabstand ist, sonst False
            float: Aktuelle Entfernung zum Tor in mm
        """
        # Tor erkennen
        gate_found, _, distance, _ = self.detect_gate()
        
        if gate_found:
            # Prüfen, ob wir im Parkabstand sind (mit Toleranz von ±20mm)
            if abs(distance - self.parkabstand_mm) < 20:
                print(f"Fahrzeug ist im Parkabstand ({self.parkabstand_mm}mm) zum Tor")
                return True, distance
            else:
                print(f"Fahrzeug ist nicht im Parkabstand: {distance:.0f}mm vs. {self.parkabstand_mm}mm")
                return False, distance
        
        return False, 0
    
    def approach_parking_position(self, speed=10):
        """
        Fährt zum Parkabstand vor dem Tor
        
        Args:
            speed: Fahrgeschwindigkeit
            
        Returns:
            bool: True wenn Parkposition erreicht, sonst False
        """
        max_attempts = 30
        attempts = 0
        
        while attempts < max_attempts:
            # Tor erkennen
            gate_found, position, distance, _ = self.detect_gate()
            
            if gate_found:
                # Lenkung anpassen, um zum Tor zu fahren
                steering_angle = -position * 25  # Proportionale Steuerung
                steering_angle = max(-35, min(35, steering_angle))
                self.px.set_dir_servo_angle(steering_angle)
                
                # Entfernung zum Parkabstand berechnen
                distance_to_parking = distance - self.parkabstand_mm
                
                if abs(distance_to_parking) < 20:
                    # Wir sind im Parkabstand, anhalten
                    self.px.stop()
                    print(f"Parkposition erreicht: {distance:.0f}mm vom Tor")
                    return True
                elif distance_to_parking > 0:
                    # Wir sind zu weit weg, vorwärts fahren
                    # Geschwindigkeit basierend auf Entfernung anpassen
                    adjusted_speed = min(speed, max(3, speed * distance_to_parking / 300))
                    self.px.forward(adjusted_speed)
                    print(f"Fahre vorwärts zum Parkabstand: {distance:.0f}mm, noch {distance_to_parking:.0f}mm")
                else:
                    # Wir sind zu nah, rückwärts fahren
                    adjusted_speed = min(speed, max(3, speed * abs(distance_to_parking) / 300))
                    self.px.backward(adjusted_speed)
                    print(f"Fahre rückwärts zum Parkabstand: {distance:.0f}mm, {abs(distance_to_parking):.0f}mm zu nah")
            else:
                # Wenn kein Tor erkannt, anhalten
                self.px.stop()
                print("Kein Tor erkannt, kann Parkposition nicht anfahren")
                return False
            
            time.sleep(0.1)
            attempts += 1
        
        self.px.stop()
        print("Konnte Parkposition nicht erreichen")
        return False
    
    def calibrate_for_small_gate(self):
        """
        Kalibriert die Erkennung für ein schmales Tor
        
        Returns:
            bool: True wenn Kalibrierung erfolgreich, sonst False
        """
        print("Kalibriere für schmales Tor (210mm)...")
        
        # Mehrere Messungen durchführen
        measurements = []
        for _ in range(10):
            gate_found, _, distance, _ = self.detect_gate()
            if gate_found:
                measurements.append(distance)
            time.sleep(0.2)
        
        if len(measurements) >= 5:
            # Median der Messungen berechnen
            median_distance = np.median(measurements)
            
            # Kalibrierungsfaktor anpassen
            if median_distance > 0:
                # Annahme: Wir sind etwa 500mm vom Tor entfernt
                self.small_object_calibration = median_distance / 500
                print(f"Kalibrierungsfaktor angepasst: {self.small_object_calibration:.2f}")
                return True
        
        print("Kalibrierung fehlgeschlagen, verwende Standardwert")
        return False
    
    def cleanup(self):
        """Aufräumen und Kamera schließen"""
        self.px.stop()
        try:
            Vilib.camera_close()
        except:
            pass
        print("Torerkennung beendet")

# Beispiel für die Verwendung
if __name__ == "__main__":
    try:
        px = Picarx()
        gate_detector = TorErkennung(px)
        
        print("Starte Torerkennung...")
        
        # Kalibrierung für schmales Tor
        gate_detector.calibrate_for_small_gate()
        
        # Tor erkennen und Informationen ausgeben
        for _ in range(5):
            gate_detector.detect_gate()
            time.sleep(0.5)
        
        # Zum Tor ausrichten
        gate_detector.align_with_gate()
        
        # Zur Parkposition fahren
        if gate_detector.approach_parking_position():
            print("Parkposition erfolgreich erreicht")
        else:
            print("Konnte Parkposition nicht erreichen")
        
        print("Torerkennung beendet")
        
    except KeyboardInterrupt:
        print("Programm durch Benutzer unterbrochen")
    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        if 'gate_detector' in locals():
            gate_detector.cleanup()
