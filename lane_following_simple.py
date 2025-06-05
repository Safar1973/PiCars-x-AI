#!/usr/bin/env python3

from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from time import sleep, time, strftime, localtime
import numpy as np
import cv2
import os
import sys

# Reset MCU und Initialisierung
reset_mcu()
sleep(0.2)

# Picarx-Objekt erstellen
px = Picarx()

# Konfigurationswerte
FORWARD_DISTANCE = 250  # 2,5 Meter in cm
SPEED_FORWARD = 20     # Reduzierte Geschwindigkeit für bessere Kontrolle
SPEED_BACKWARD = 15
TURN_ANGLE = 180       # Drehwinkel in Grad
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Status-Variablen
line_detected = False
center_position = 0
distance_traveled = 0
start_time = 0

def setup_camera():
    """Kamera initialisieren und Bildverarbeitung starten"""
    print("Kamera wird initialisiert...")
    try:
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=False)
        
        # Einfache Farberkennung aktivieren, ohne problematische Funktionen
        try:
            # Verwenden Sie eine Standardfarbe, die in Vilib bereits definiert ist
            Vilib.color_detect("red")
            print("Farberkennung aktiviert.")
        except Exception as e:
            print(f"Fehler bei der Aktivierung der Farberkennung: {e}")
        
        sleep(2)  # Warten auf Kamera-Initialisierung
        print("Kamera bereit!")
        return True
    except Exception as e:
        print(f"Fehler bei der Kamera-Initialisierung: {e}")
        return False

def detect_lane_opencv():
    """Fahrspurbegrenzungen mit OpenCV erkennen"""
    global line_detected, center_position
    
    try:
        # Bild von Vilib holen, falls verfügbar
        if hasattr(Vilib, 'img') and Vilib.img is not None:
            frame = Vilib.img.copy()
            
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Bild weichzeichnen
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Kanten erkennen
            edges = cv2.Canny(blur, 50, 150)
            
            # Untere Hälfte des Bildes für die Linienerkennung verwenden
            height, width = edges.shape
            roi = edges[height//2:height, 0:width]
            
            # Linien erkennen mit Hough-Transformation
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
                if len(left_lines) > 0 or len(right_lines) > 0:
                    left_x = 0
                    right_x = width
                    
                    # Durchschnitt der linken Linien berechnen
                    if len(left_lines) > 0:
                        left_x_sum = sum([x1 for x1, _, _, _ in left_lines])
                        left_x = left_x_sum / len(left_lines)
                    
                    # Durchschnitt der rechten Linien berechnen
                    if len(right_lines) > 0:
                        right_x_sum = sum([x1 for x1, _, _, _ in right_lines])
                        right_x = right_x_sum / len(right_lines)
                    
                    # Mittelpunkt zwischen den Linien berechnen
                    center_x = (left_x + right_x) / 2
                    
                    # Position relativ zur Bildmitte (-1 bis 1)
                    center_position = (center_x - width/2) / (width/2)
                    line_detected = True
                    
                    print(f"Fahrspurbegrenzungen erkannt: Position = {center_position:.2f}")
                    return True, center_position
            
            line_detected = False
            return False, 0
        else:
            print("Kein Bild verfügbar für OpenCV-Verarbeitung")
            line_detected = False
            return False, 0
    except Exception as e:
        print(f"Fehler bei der OpenCV-Linienerkennung: {e}")
        line_detected = False
        return False, 0

def detect_lane():
    """Fahrspurbegrenzungen erkennen"""
    # OpenCV-Methode zur Erkennung beider Fahrspurbegrenzungen
    return detect_lane_opencv()

def adjust_steering(is_backward=False):
    """Lenkung basierend auf der Position zwischen den Fahrspurbegrenzungen anpassen"""
    global center_position
    
    try:
        if line_detected:
            # Lenkwinkel basierend auf der Position berechnen
            # Multiplikator kann angepasst werden, um Empfindlichkeit zu steuern
            multiplier = 30  # Empfindlichkeit der Lenkung
            
            if is_backward:
                # Invertierte Lenkung für Rückwärtsfahrt
                steering_angle = multiplier * center_position
            else:
                # Normale Lenkung für Vorwärtsfahrt
                steering_angle = -multiplier * center_position
            
            # Lenkwinkel begrenzen
            steering_angle = max(-30, min(30, steering_angle))
            
            # Lenkwinkel setzen
            px.set_dir_servo_angle(steering_angle)
            
            direction = "rückwärts" if is_backward else "vorwärts"
            print(f"Lenkwinkel angepasst ({direction}): {steering_angle:.1f}°")
            return True
        else:
            # Wenn keine Fahrspurbegrenzungen erkannt, geradeaus fahren
            px.set_dir_servo_angle(0)
            print("Keine Fahrspurbegrenzungen erkannt, fahre geradeaus")
            return False
    except Exception as e:
        print(f"Fehler bei der Lenkungsanpassung: {e}")
        # Sicherheitshalber geradeaus fahren
        px.set_dir_servo_angle(0)
        return False

def emergency_stop():
    """Notfall-Stopp bei kritischen Situationen"""
    print("NOTFALL-STOPP AKTIVIERT!")
    px.stop()
    sleep(1)

def drive_forward(distance_cm, speed=20):
    """Zwischen den Fahrspurbegrenzungen für eine bestimmte Distanz fahren"""
    print(f"Fahre {distance_cm} cm vorwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 10 cm/s bei Geschwindigkeit 20
    estimated_time = distance_cm / (speed / 5)
    
    # Vorwärts fahren starten
    px.forward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    lane_lost_count = 0
    max_lane_lost = 20  # Maximale Anzahl von Frames, in denen die Fahrspurbegrenzungen verloren sein dürfen
    
    while elapsed_time < estimated_time:
        # Fahrspurbegrenzungen erkennen
        lane_found, _ = detect_lane()
        
        if lane_found:
            # Lenkung anpassen
            adjust_steering(is_backward=False)
            lane_lost_count = 0  # Zurücksetzen des Zählers
        else:
            lane_lost_count += 1
            print(f"Fahrspurbegrenzungen verloren ({lane_lost_count}/{max_lane_lost})")
            
            # Wenn die Fahrspurbegrenzungen zu lange verloren sind, langsamer werden
            if lane_lost_count > 5:
                px.forward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Fahrspurbegrenzungen zu lange verloren sind, abbrechen
            if lane_lost_count > max_lane_lost:
                print("Fahrspurbegrenzungen dauerhaft verloren, stoppe...")
                px.stop()
                return False
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.05)  # Kürzere Wartezeit für schnellere Reaktion
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Vorwärtsbewegung abgeschlossen")
    return True

def turn(angle, speed=15):
    """Drehen um einen bestimmten Winkel"""
    print(f"Drehe um {angle}°...")
    
    try:
        # Richtung bestimmen
        direction = 1 if angle > 0 else -1
        angle = abs(angle)
        
        # Geschätzte Zeit für die Drehung berechnen
        # Annahme: 90° bei Geschwindigkeit 15 dauert ca. 2.5 Sekunden
        estimated_time = (angle / 90) * 2.5
        
        # Drehung starten
        if direction > 0:
            px.set_dir_servo_angle(30)  # Maximaler Lenkwinkel
            px.forward(speed)
        else:
            px.set_dir_servo_angle(-30)  # Maximaler Lenkwinkel in die andere Richtung
            px.forward(speed)
        
        # Warten bis Drehung abgeschlossen
        sleep(estimated_time)
        
        # Anhalten nach Drehung
        px.stop()
        px.set_dir_servo_angle(0)
        
        # Kurze Pause, um die Kamera zu stabilisieren
        sleep(0.5)
        
        print("Drehung abgeschlossen")
        return True
    except Exception as e:
        print(f"Fehler bei der Drehung: {e}")
        emergency_stop()
        return False

def drive_backward(distance_cm, speed=15):
    """Zwischen den Fahrspurbegrenzungen rückwärts für eine bestimmte Distanz fahren"""
    print(f"Fahre {distance_cm} cm rückwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 8 cm/s bei Geschwindigkeit 15
    estimated_time = distance_cm / (speed / 5)
    
    # Rückwärts fahren starten
    px.backward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    lane_lost_count = 0
    max_lane_lost = 20  # Maximale Anzahl von Frames, in denen die Fahrspurbegrenzungen verloren sein dürfen
    
    while elapsed_time < estimated_time:
        # Fahrspurbegrenzungen erkennen
        lane_found, _ = detect_lane()
        
        if lane_found:
            # Lenkung anpassen (invertiert für Rückwärtsfahrt)
            adjust_steering(is_backward=True)
            lane_lost_count = 0  # Zurücksetzen des Zählers
        else:
            lane_lost_count += 1
            print(f"Fahrspurbegrenzungen verloren ({lane_lost_count}/{max_lane_lost})")
            
            # Wenn die Fahrspurbegrenzungen zu lange verloren sind, langsamer werden
            if lane_lost_count > 5:
                px.backward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Fahrspurbegrenzungen zu lange verloren sind, abbrechen
            if lane_lost_count > max_lane_lost:
                print("Fahrspurbegrenzungen dauerhaft verloren, stoppe...")
                px.stop()
                return False
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.05)  # Kürzere Wartezeit für schnellere Reaktion
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Rückwärtsbewegung abgeschlossen")
    return True

def take_photo():
    """Foto aufnehmen und speichern"""
    try:
        _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
        name = 'lane_following_%s' % _time
        user = os.getlogin()
        user_home = os.path.expanduser(f'~{user}')
        path = f"{user_home}/Pictures/picar-x/"
        
        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(path, exist_ok=True)
        
        Vilib.take_photo(name, path)
        print(f'Foto gespeichert als {path}{name}.jpg')
        return True
    except Exception as e:
        print(f"Fehler beim Aufnehmen des Fotos: {e}")
        return False

def check_system():
    """Systemprüfung vor dem Start"""
    print("Führe Systemprüfung durch...")
    
    # Prüfe Kamera
    if not setup_camera():
        print("FEHLER: Kamera konnte nicht initialisiert werden.")
        return False
    
    # Prüfe Motoren durch kurze Bewegung
    try:
        px.forward(10)
        sleep(0.5)
        px.stop()
        sleep(0.5)
        px.backward(10)
        sleep(0.5)
        px.stop()
        print("Motortest erfolgreich.")
    except Exception as e:
        print(f"FEHLER: Motortest fehlgeschlagen: {e}")
        return False
    
    # Prüfe Lenkservo
    try:
        px.set_dir_servo_angle(20)
        sleep(0.5)
        px.set_dir_servo_angle(-20)
        sleep(0.5)
        px.set_dir_servo_angle(0)
        print("Servotest erfolgreich.")
    except Exception as e:
        print(f"FEHLER: Servotest fehlgeschlagen: {e}")
        return False
    
    print("Systemprüfung abgeschlossen. Alle Systeme funktionieren.")
    return True

def main():
    """Hauptfunktion für die autonome Fahrt"""
    try:
        print("=== Autonomes Fahren zwischen Fahrspurbegrenzungen wird gestartet ===")
        
        # Systemprüfung
        if not check_system():
            print("Systemprüfung fehlgeschlagen. Programm wird beendet.")
            return
        
        # Foto vom Ausgangspunkt machen
        try:
            take_photo()
        except Exception as e:
            print(f"Hinweis: Foto konnte nicht aufgenommen werden: {e}")
        
        # 1. Vorwärts zwischen den Fahrspurbegrenzungen fahren (2,5 Meter)
        print("\n=== Schritt 1: 2,5 Meter vorwärts fahren ===")
        if not drive_forward(FORWARD_DISTANCE, SPEED_FORWARD):
            print("Vorwärtsfahrt konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 2. Drehen (180 Grad)
        print("\n=== Schritt 2: Drehen ===")
        if not turn(TURN_ANGLE, 15):
            print("Drehung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # Nach der Drehung kurz warten und prüfen, ob die Fahrspurbegrenzungen erkannt werden
        print("Suche Fahrspurbegrenzungen nach der Drehung...")
        lane_found = False
        for _ in range(10):  # 10 Versuche
            lane_found, _ = detect_lane()
            if lane_found:
                print("Fahrspurbegrenzungen nach der Drehung erkannt!")
                break
            sleep(0.5)
        
        if not lane_found:
            print("WARNUNG: Keine Fahrspurbegrenzungen nach der Drehung erkannt. Versuche trotzdem fortzufahren.")
        
        # 3. Rückwärts zwischen den Fahrspurbegrenzungen fahren (zurück zum Ausgangspunkt)
        print("\n=== Schritt 3: 2,5 Meter rückwärts fahren ===")
        if not drive_backward(FORWARD_DISTANCE, SPEED_BACKWARD):
            print("Rückwärtsfahrt konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # Foto vom Endpunkt machen
        try:
            take_photo()
        except Exception as e:
            print(f"Hinweis: Foto konnte nicht aufgenommen werden: {e}")
        
        print("\n=== Autonome Fahrt erfolgreich abgeschlossen! ===")
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer unterbrochen.")
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        print("Programm wird beendet.")
    finally:
        # Aufräumen
        px.stop()
        try:
            Vilib.camera_close()
        except:
            pass
        print("Programm beendet.")

if __name__ == "__main__":
    main()
