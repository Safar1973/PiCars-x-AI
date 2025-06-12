#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linienverfolgung.py - Modul für die robuste Erkennung und Verfolgung von zwei roten Linien
"""

from picarx import Picarx
import cv2
import numpy as np
import time
from vilib import Vilib

class ZweiLinienVerfolger:
    def __init__(self, px):
        """
        Initialisiert den Zwei-Linien-Verfolger
        
        Args:
            px: Picarx-Objekt
        """
        self.px = px
        self.camera_width = 320
        self.camera_height = 240
        
        # Parameter für die rote Linienerkennung (HSV-Werte für rot)
        # Rot hat zwei Bereiche im HSV-Farbraum
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Parameter für die Steuerung
        self.steering_sensitivity = 0.5  # Empfindlichkeit der Lenkung
        self.base_speed = 25  # Grundgeschwindigkeit
        
        # Parameter für robuste Linienverfolgung
        self.last_center_positions = []  # Speichert die letzten gültigen Mittelpositionen
        self.max_history = 5  # Anzahl der zu speichernden Positionen
        
        # Parameter für die Streckenmessung
        self.distance_traveled = 0  # Zurückgelegte Strecke in mm
        self.last_time = None  # Zeitpunkt der letzten Messung
        self.speed_mm_per_sec = 100  # Geschätzte Geschwindigkeit in mm/s bei speed=30
        
        # Kamera initialisieren
        self._setup_camera()
        
        # Status-Variablen
        self.lines_detected = False
        self.left_line_position = -1  # Position der linken Linie (-1 bis 1)
        self.right_line_position = 1  # Position der rechten Linie (-1 bis 1)
        self.center_position = 0  # Position zwischen den Linien (-1 bis 1, 0 ist Mitte)
        self.lane_width = 0  # Breite der Fahrspur in Pixeln
        self.consecutive_misses = 0  # Zählt aufeinanderfolgende Frames ohne Linienerkennung
        self.max_consecutive_misses = 10  # Maximale Anzahl aufeinanderfolgender Frames ohne Linien
    
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
    
    def _predict_center_position(self):
        """
        Sagt die Position der Mitte basierend auf früheren Positionen voraus
        
        Returns:
            float: Vorhergesagte Position der Mitte
        """
        if not self.last_center_positions:
            return 0
        
        # Gewichteter Durchschnitt der letzten Positionen
        # Neuere Positionen haben mehr Gewicht
        weights = [i+1 for i in range(len(self.last_center_positions))]
        weighted_sum = sum(pos * w for pos, w in zip(self.last_center_positions, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def detect_lines(self):
        """
        Erkennt zwei rote Linien im Kamerabild
        
        Returns:
            bool: True wenn beide Linien erkannt wurden, sonst False
            float: Position der Mitte zwischen den Linien (-1 bis 1, 0 ist Mitte)
            float: Breite der Fahrspur in Pixeln
        """
        try:
            # Bild von Vilib holen
            if hasattr(Vilib, 'img') and Vilib.img is not None:
                frame = Vilib.img.copy()
                
                # In HSV-Farbraum konvertieren
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Masken für beide roten Farbbereiche erstellen
                mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
                mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
                
                # Masken kombinieren
                mask = cv2.bitwise_or(mask1, mask2)
                
                # Rauschen entfernen
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # Untere Hälfte des Bildes für die Linienerkennung verwenden
                height, width = mask.shape
                roi = mask[height//2:height, 0:width]
                
                # Linien mit Hough-Transformation erkennen
                lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, 
                                        minLineLength=50, maxLineGap=100)
                
                if lines is not None and len(lines) > 0:
                    # Linien in links und rechts aufteilen
                    left_lines = []
                    right_lines = []
                    
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # Steigung berechnen (um vertikale Linien zu vermeiden)
                        if x2 - x1 == 0:
                            continue
                            
                        slope = (y2 - y1) / (x2 - x1)
                        
                        # Linien nach Steigung filtern
                        if abs(slope) < 0.1:  # Fast horizontale Linien ignorieren
                            continue
                            
                        # Linien nach Position links/rechts aufteilen
                        if slope < 0 and x1 < width/2:  # Linke Linie
                            left_lines.append(line[0])
                        elif slope > 0 and x1 > width/2:  # Rechte Linie
                            right_lines.append(line[0])
                    
                    # Wenn mindestens eine Linie auf jeder Seite gefunden wurde
                    if len(left_lines) > 0 and len(right_lines) > 0:
                        # Durchschnitt der linken Linien berechnen
                        left_x_sum = sum([x1 for x1, _, _, _ in left_lines])
                        left_x = left_x_sum / len(left_lines)
                        
                        # Durchschnitt der rechten Linien berechnen
                        right_x_sum = sum([x1 for x1, _, _, _ in right_lines])
                        right_x = right_x_sum / len(right_lines)
                        
                        # Positionen relativ zur Bildmitte (-1 bis 1)
                        self.left_line_position = (left_x - width/2) / (width/2)
                        self.right_line_position = (right_x - width/2) / (width/2)
                        
                        # Mittelpunkt zwischen den Linien berechnen
                        center_x = (left_x + right_x) / 2
                        self.center_position = (center_x - width/2) / (width/2)
                        
                        # Breite der Fahrspur in Pixeln
                        self.lane_width = right_x - left_x
                        
                        # Position in Historie speichern
                        self.last_center_positions.append(self.center_position)
                        if len(self.last_center_positions) > self.max_history:
                            self.last_center_positions.pop(0)
                        
                        self.lines_detected = True
                        self.consecutive_misses = 0  # Zurücksetzen des Zählers
                        
                        print(f"Zwei rote Linien erkannt: Mitte = {self.center_position:.2f}, Breite = {self.lane_width:.0f}px")
                        return True, self.center_position, self.lane_width
                
                # Wenn keine zwei Linien erkannt wurden
                self.consecutive_misses += 1
                
                if self.consecutive_misses <= self.max_consecutive_misses and self.last_center_positions:
                    # Verwende die letzte bekannte Position
                    predicted_position = self._predict_center_position()
                    print(f"Linien kurzzeitig verloren, verwende Vorhersage: {predicted_position:.2f}")
                    
                    # Wir geben an, dass die Linien noch "erkannt" sind, verwenden aber die Vorhersage
                    self.center_position = predicted_position
                    return True, predicted_position, self.lane_width
                else:
                    self.lines_detected = False
                    print(f"Keine zwei roten Linien erkannt (Versuche: {self.consecutive_misses})")
                    return False, 0, 0
            else:
                print("Kein Bild verfügbar")
                self.lines_detected = False
                return False, 0, 0
        except Exception as e:
            print(f"Fehler bei der Linienerkennung: {e}")
            self.lines_detected = False
            return False, 0, 0
    
    def adjust_steering(self):
        """
        Passt die Lenkung basierend auf der Position zwischen den Linien an
        
        Returns:
            bool: True wenn Lenkung angepasst wurde, sonst False
        """
        if self.lines_detected:
            # Lenkwinkel basierend auf der Position berechnen
            steering_angle = -self.steering_sensitivity * self.center_position * 35
            
            # Lenkwinkel begrenzen
            steering_angle = max(-35, min(35, steering_angle))
            
            # Lenkwinkel setzen
            self.px.set_dir_servo_angle(steering_angle)
            
            print(f"Lenkwinkel angepasst: {steering_angle:.1f}°")
            return True
        else:
            # Wenn keine Linien erkannt, geradeaus fahren
            self.px.set_dir_servo_angle(0)
            print("Keine Linien erkannt, fahre geradeaus")
            return False
    
    def reset_distance(self):
        """Setzt die zurückgelegte Strecke zurück"""
        self.distance_traveled = 0
        self.last_time = time.time()
        print("Streckenzähler zurückgesetzt")
    
    def update_distance(self, speed):
        """
        Aktualisiert die zurückgelegte Strecke basierend auf der Zeit und Geschwindigkeit
        
        Args:
            speed: Aktuelle Geschwindigkeit (0-100)
            
        Returns:
            float: Zurückgelegte Strecke in mm
        """
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            return self.distance_traveled
        
        # Zeitdifferenz berechnen
        time_diff = current_time - self.last_time
        
        # Geschwindigkeit in mm/s basierend auf der Geschwindigkeitseinstellung
        speed_mm_per_sec = (speed / 30) * self.speed_mm_per_sec
        
        # Zurückgelegte Strecke berechnen
        distance_increment = speed_mm_per_sec * time_diff
        self.distance_traveled += distance_increment
        
        # Zeit aktualisieren
        self.last_time = current_time
        
        return self.distance_traveled
    
    def is_target_distance_reached(self, target_distance):
        """
        Prüft, ob die Zielstrecke erreicht wurde
        
        Args:
            target_distance: Zielstrecke in mm
            
        Returns:
            bool: True wenn die Zielstrecke erreicht wurde, sonst False
        """
        return self.distance_traveled >= target_distance
    
    def follow_lines(self, speed=None):
        """
        Folgt den beiden roten Linien mit der angegebenen Geschwindigkeit
        
        Args:
            speed: Geschwindigkeit (wenn None, wird base_speed verwendet)
        
        Returns:
            bool: True wenn Linien erkannt und gefolgt wurde, sonst False
        """
        if speed is None:
            speed = self.base_speed
        
        # Linien erkennen
        lines_found, _, _ = self.detect_lines()
        
        if lines_found:
            # Lenkung anpassen
            self.adjust_steering()
            
            # Geschwindigkeit basierend auf Lenkwinkel anpassen
            # Bei starken Kurven langsamer fahren
            angle = abs(self.px.dir_current_angle)
            if angle > 25:
                adjusted_speed = speed * 0.7
            elif angle > 15:
                adjusted_speed = speed * 0.85
            else:
                adjusted_speed = speed
            
            # Vorwärts fahren
            self.px.forward(adjusted_speed)
            
            # Zurückgelegte Strecke aktualisieren
            self.update_distance(adjusted_speed)
            
            return True
        else:
            # Wenn keine Linien erkannt, langsamer fahren und suchen
            self.px.forward(speed * 0.5)
            
            # Suchbewegung durchführen
            if self.consecutive_misses % 2 == 0:
                self.px.set_dir_servo_angle(10)  # Nach rechts suchen
            else:
                self.px.set_dir_servo_angle(-10)  # Nach links suchen
            
            # Zurückgelegte Strecke aktualisieren (mit reduzierter Geschwindigkeit)
            self.update_distance(speed * 0.5)
            
            return False
    
    def follow_lines_for_distance(self, distance_mm, speed=None):
        """
        Folgt den beiden roten Linien für eine bestimmte Strecke
        
        Args:
            distance_mm: Zu fahrende Strecke in mm
            speed: Geschwindigkeit (wenn None, wird base_speed verwendet)
            
        Returns:
            bool: True wenn die Strecke erfolgreich zurückgelegt wurde, sonst False
        """
        if speed is None:
            speed = self.base_speed
        
        # Streckenzähler zurücksetzen
        self.reset_distance()
        
        print(f"Starte Linienverfolgung für {distance_mm}mm")
        
        # Fahren, bis die Zielstrecke erreicht ist
        while not self.is_target_distance_reached(distance_mm):
            lines_found = self.follow_lines(speed)
            
            if not lines_found:
                self.consecutive_misses += 1
                print(f"Linien verloren ({self.consecutive_misses}/{self.max_consecutive_misses})")
                
                # Wenn die Linien zu lange verloren sind, abbrechen
                if self.consecutive_misses > self.max_consecutive_misses:
                    print("Linien dauerhaft verloren, breche ab")
                    self.px.stop()
                    return False
            else:
                self.consecutive_misses = 0
            
            time.sleep(0.05)
        
        # Anhalten nach Erreichen der Strecke
        self.px.stop()
        print(f"Zielstrecke von {distance_mm}mm erreicht")
        return True
    
    def turn_180_between_lines(self, speed=15):
        """
        Dreht das Fahrzeug um 180 Grad zwischen den beiden roten Linien
        
        Args:
            speed: Drehgeschwindigkeit
            
        Returns:
            bool: True wenn Drehung erfolgreich, sonst False
        """
        print("Drehe um 180 Grad zwischen den Linien...")
        
        # Startwinkel merken
        start_angle = self.px.dir_current_angle
        
        # Maximaler Lenkwinkel für die Drehung
        self.px.set_dir_servo_angle(35)
        
        # Vorwärts fahren, um zu drehen
        self.px.forward(speed)
        
        # Drehung durchführen und dabei Linien überwachen
        start_time = time.time()
        rotation_complete = False
        consecutive_misses = 0
        max_rotation_time = 10  # Maximale Zeit für die Drehung in Sekunden
        
        while time.time() - start_time < max_rotation_time and not rotation_complete:
            # Linien erkennen
            lines_found, _, _ = self.detect_lines()
            
            if not lines_found:
                consecutive_misses += 1
                print(f"Linien während der Drehung verloren ({consecutive_misses}/{self.max_consecutive_misses})")
                
                # Wenn die Linien zu lange verloren sind, abbrechen
                if consecutive_misses > self.max_consecutive_misses:
                    print("Linien während der Drehung dauerhaft verloren, breche ab")
                    self.px.stop()
                    return False
            else:
                consecutive_misses = 0
            
            # Prüfen, ob die Drehung abgeschlossen ist (ca. 180 Grad)
            # Da wir keinen Kompass haben, schätzen wir die Drehung anhand der Zeit
            if time.time() - start_time > 5.0:  # Empirisch bestimmte Zeit für 180 Grad
                rotation_complete = True
            
            time.sleep(0.05)
        
        # Anhalten und Lenkung zurücksetzen
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        
        print("Drehung abgeschlossen")
        time.sleep(1)  # Kurze Pause zur Stabilisierung
        
        return rotation_complete
    
    def park_between_lines(self, speed=15):
        """
        Parkt das Fahrzeug zwischen den beiden roten Linien
        
        Args:
            speed: Fahrgeschwindigkeit
            
        Returns:
            bool: True wenn Parken erfolgreich, sonst False
        """
        print("Parke zwischen den roten Linien...")
        
        # Langsam vorwärts fahren und Linien überwachen
        self.px.forward(speed)
        
        # Zähler für stabile Positionierung
        stable_position_count = 0
        required_stable_positions = 10  # Anzahl der Frames mit stabiler Position
        
        # Parken durchführen und dabei Linien überwachen
        start_time = time.time()
        max_parking_time = 20  # Maximale Zeit für das Parken in Sekunden
        
        while time.time() - start_time < max_parking_time:
            # Linien erkennen
            lines_found, center_position, lane_width = self.detect_lines()
            
            if lines_found:
                # Lenkung anpassen
                self.adjust_steering()
                
                # Prüfen, ob wir mittig zwischen den Linien sind
                if abs(center_position) < 0.1:
                    stable_position_count += 1
                    print(f"Stabile Parkposition: {stable_position_count}/{required_stable_positions}")
                    
                    # Wenn wir lange genug stabil sind, parken abschließen
                    if stable_position_count >= required_stable_positions:
                        self.px.stop()
                        print("Fahrzeug erfolgreich zwischen den Linien geparkt")
                        return True
                else:
                    stable_position_count = 0
            else:
                self.consecutive_misses += 1
                print(f"Linien während des Parkens verloren ({self.consecutive_misses}/{self.max_consecutive_misses})")
                
                # Wenn die Linien zu lange verloren sind, abbrechen
                if self.consecutive_misses > self.max_consecutive_misses:
                    print("Linien während des Parkens dauerhaft verloren, breche ab")
                    self.px.stop()
                    return False
            
            time.sleep(0.05)
        
        # Zeit abgelaufen
        self.px.stop()
        print("Zeitlimit für das Parken überschritten")
        return False
    
    def stop(self):
        """Stoppt das Fahrzeug"""
        self.px.stop()
        print("Fahrzeug angehalten")
    
    def cleanup(self):
        """Aufräumen und Kamera schließen"""
        self.px.stop()
        try:
            Vilib.camera_close()
        except:
            pass
        print("Linienverfolger beendet")

# Beispiel für die Verwendung
if __name__ == "__main__":
    try:
        px = Picarx()
        line_follower = ZweiLinienVerfolger(px)
        
        print("Starte Test der Linienverfolgung...")
        
        # Den beiden roten Linien für 30cm folgen
        if line_follower.follow_lines_for_distance(300, 25):
            print("30cm erfolgreich zurückgelegt!")
            
            # 180-Grad-Drehung zwischen den Linien
            if line_follower.turn_180_between_lines(15):
                print("180-Grad-Drehung erfolgreich!")
                
                # Den beiden roten Linien für 2m folgen
                if line_follower.follow_lines_for_distance(2000, 25):
                    print("2m erfolgreich zurückgelegt!")
                    
                    # Zwischen den Linien parken
                    if line_follower.park_between_lines(15):
                        print("Erfolgreich zwischen den Linien geparkt!")
                    else:
                        print("Parken fehlgeschlagen")
                else:
                    print("Konnte 2m nicht erfolgreich zurücklegen")
            else:
                print("180-Grad-Drehung fehlgeschlagen")
        else:
            print("Konnte 30cm nicht erfolgreich zurücklegen")
        
        print("Test beendet")
        
    except KeyboardInterrupt:
        print("Programm durch Benutzer unterbrochen")
    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        if 'line_follower' in locals():
            line_follower.cleanup()
