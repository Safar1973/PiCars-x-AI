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
SPEED_FORWARD = 25     # Reduzierte Geschwindigkeit für bessere Kontrolle
SPEED_BACKWARD = 20
TURN_ANGLE = 180       # Drehwinkel in Grad
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
LINE_COLOR = "brown"   # Braune Linie

# Status-Variablen
line_detected = False
line_position = 0
distance_traveled = 0
start_time = 0

# HSV-Farbbereich für Braun definieren
# Diese Werte müssen möglicherweise angepasst werden
BROWN_LOWER = np.array([10, 100, 20])   # Unterer HSV-Bereich für Braun
BROWN_UPPER = np.array([30, 255, 200])  # Oberer HSV-Bereich für Braun

def setup_camera():
    """Kamera initialisieren und Bildverarbeitung starten"""
    print("Kamera wird initialisiert...")
    try:
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=True, web=False)
        
        # Eigene Farbdefinition für Braun hinzufügen
        try:
            # Versuchen, die Farbe Braun zu definieren
            # Dies ist ein Workaround, da die Standardfarben in Vilib nicht Braun enthalten
            if hasattr(Vilib, 'color_dict'):
                Vilib.color_dict['brown'] = [[10, 30], [100, 255], [20, 200]]
                print("Braune Farbe erfolgreich definiert.")
            else:
                print("Warnung: color_dict nicht verfügbar, verwende Standard-Farberkennung.")
        except Exception as e:
            print(f"Fehler beim Definieren der braunen Farbe: {e}")
        
        # Linienerkennung aktivieren
        Vilib.color_detect("brown")  # Versuchen, die braune Farbe zu erkennen
        
        sleep(2)  # Warten auf Kamera-Initialisierung
        print("Kamera bereit!")
        return True
    except Exception as e:
        print(f"Fehler bei der Kamera-Initialisierung: {e}")
        return False

def detect_line_opencv():
    """Braune Linie mit OpenCV erkennen (Fallback-Methode)"""
    global line_detected, line_position
    
    try:
        # Bild von Vilib holen, falls verfügbar
        if hasattr(Vilib, 'img') and Vilib.img is not None:
            frame = Vilib.img.copy()
            
            # Bild in HSV-Farbraum konvertieren
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Maske für braune Farbe erstellen
            mask = cv2.inRange(hsv, BROWN_LOWER, BROWN_UPPER)
            
            # Rauschen reduzieren
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Konturen finden
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Wenn Konturen gefunden wurden
            if len(contours) > 0:
                # Größte Kontur finden
                c = max(contours, key=cv2.contourArea)
                
                # Schwerpunkt der Kontur berechnen
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    
                    # Position relativ zur Bildmitte (-1 bis 1)
                    line_position = (cx - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
                    line_detected = True
                    
                    # Debug-Ausgabe
                    print(f"Linie erkannt (OpenCV): Position = {line_position:.2f}")
                    
                    return True, line_position
            
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

def detect_line():
    """Linie erkennen mit Vilib oder OpenCV als Fallback"""
    global line_detected, line_position
    
    try:
        # Zuerst versuchen, mit Vilib zu erkennen
        if hasattr(Vilib, 'color_detect_result') and Vilib.color_detect_result is not None:
            # Prüfen, ob Farbobjekte erkannt wurden
            if Vilib.color_detect_result.get('color_n', 0) > 0:
                line_detected = True
                
                # Position der Linie im Bild (x-Koordinate)
                line_x = Vilib.color_detect_result.get('x', CAMERA_WIDTH/2)
                
                # Abweichung von der Bildmitte berechnen (-1 bis 1)
                line_position = (line_x - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
                
                print(f"Linie erkannt (Vilib): Position = {line_position:.2f}")
                return True, line_position
        
        # Wenn Vilib keine Linie erkennt, OpenCV als Fallback verwenden
        return detect_line_opencv()
    
    except Exception as e:
        print(f"Fehler bei der Linienerkennung: {e}")
        # Bei Fehler OpenCV als Fallback verwenden
        return detect_line_opencv()

def adjust_steering():
    """Lenkung basierend auf Linienposition anpassen"""
    global line_position
    
    try:
        if line_detected:
            # Lenkwinkel basierend auf Linienposition berechnen
            # Multiplikator kann angepasst werden, um Empfindlichkeit zu steuern
            steering_angle = -30 * line_position
            
            # Lenkwinkel begrenzen
            steering_angle = max(-30, min(30, steering_angle))
            
            # Lenkwinkel setzen
            px.set_dir_servo_angle(steering_angle)
            print(f"Lenkwinkel angepasst: {steering_angle:.1f}°")
            return True
        else:
            # Wenn keine Linie erkannt, geradeaus fahren
            px.set_dir_servo_angle(0)
            print("Keine Linie erkannt, fahre geradeaus")
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

def follow_line_forward(distance_cm, speed=25):
    """Der Linie folgen für eine bestimmte Distanz"""
    print(f"Folge der Linie {distance_cm} cm vorwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 10 cm/s bei Geschwindigkeit 25
    estimated_time = distance_cm / (speed / 4)
    
    # Vorwärts fahren starten
    px.forward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    line_lost_count = 0
    max_line_lost = 20  # Maximale Anzahl von Frames, in denen die Linie verloren sein darf
    
    while elapsed_time < estimated_time:
        # Linie erkennen
        line_found, _ = detect_line()
        
        if line_found:
            # Lenkung anpassen
            adjust_steering()
            line_lost_count = 0  # Zurücksetzen des Zählers
        else:
            line_lost_count += 1
            print(f"Linie verloren ({line_lost_count}/{max_line_lost})")
            
            # Wenn die Linie zu lange verloren ist, langsamer werden
            if line_lost_count > 5:
                px.forward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Linie zu lange verloren ist, abbrechen
            if line_lost_count > max_line_lost:
                print("Linie dauerhaft verloren, stoppe...")
                px.stop()
                return False
        
        # Zeit aktualisieren
        elapsed_time = time() - start_time
        sleep(0.05)  # Kürzere Wartezeit für schnellere Reaktion
    
    # Anhalten nach Erreichen der Distanz
    px.stop()
    print("Vorwärtsbewegung abgeschlossen")
    return True

def turn(angle, speed=20):
    """Drehen um einen bestimmten Winkel"""
    print(f"Drehe um {angle}°...")
    
    try:
        # Richtung bestimmen
        direction = 1 if angle > 0 else -1
        angle = abs(angle)
        
        # Geschätzte Zeit für die Drehung berechnen
        # Annahme: 90° bei Geschwindigkeit 20 dauert ca. 2 Sekunden
        estimated_time = (angle / 90) * 2
        
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
        print("Drehung abgeschlossen")
        return True
    except Exception as e:
        print(f"Fehler bei der Drehung: {e}")
        emergency_stop()
        return False

def follow_line_backward(distance_cm, speed=20):
    """Der Linie rückwärts folgen für eine bestimmte Distanz"""
    print(f"Folge der Linie {distance_cm} cm rückwärts...")
    
    # Startzeit merken
    start_time = time()
    
    # Geschätzte Zeit für die Distanz berechnen
    # Annahme: 8 cm/s bei Geschwindigkeit 20
    estimated_time = distance_cm / (speed / 4)
    
    # Rückwärts fahren starten
    px.backward(speed)
    
    # Fahren bis Distanz erreicht
    elapsed_time = 0
    line_lost_count = 0
    max_line_lost = 20  # Maximale Anzahl von Frames, in denen die Linie verloren sein darf
    
    while elapsed_time < estimated_time:
        # Linie erkennen
        line_found, _ = detect_line()
        
        if line_found:
            # Lenkung anpassen (invertiert für Rückwärtsfahrt)
            if line_detected:
                steering_angle = 30 * line_position  # Invertiert für Rückwärtsfahrt
                steering_angle = max(-30, min(30, steering_angle))
                px.set_dir_servo_angle(steering_angle)
                print(f"Lenkwinkel angepasst (rückwärts): {steering_angle:.1f}°")
                line_lost_count = 0  # Zurücksetzen des Zählers
        else:
            line_lost_count += 1
            print(f"Linie verloren ({line_lost_count}/{max_line_lost})")
            
            # Wenn die Linie zu lange verloren ist, langsamer werden
            if line_lost_count > 5:
                px.backward(speed * 0.7)  # Reduzierte Geschwindigkeit
            
            # Wenn die Linie zu lange verloren ist, abbrechen
            if line_lost_count > max_line_lost:
                print("Linie dauerhaft verloren, stoppe...")
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
        name = 'line_follower_%s' % _time
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
    
    # Prüfe Linienerkennung
    print("Prüfe Linienerkennung...")
    line_found = False
    for _ in range(10):  # 10 Versuche
        line_found, _ = detect_line()
        if line_found:
            print("Linie erkannt!")
            break
        sleep(0.5)
    
    if not line_found:
        print("WARNUNG: Keine Linie erkannt. Stellen Sie sicher, dass die braune Linie sichtbar ist.")
        # Trotzdem fortfahren, da die Linie möglicherweise später erkannt wird
    
    print("Systemprüfung abgeschlossen. Alle Systeme funktionieren.")
    return True

def main():
    """Hauptfunktion für die autonome Fahrt"""
    try:
        print("=== Autonomes Linienfolgen wird gestartet ===")
        
        # Systemprüfung
        if not check_system():
            print("Systemprüfung fehlgeschlagen. Programm wird beendet.")
            return
        
        # Foto vom Ausgangspunkt machen
        try:
            take_photo()
        except Exception as e:
            print(f"Hinweis: Foto konnte nicht aufgenommen werden: {e}")
        
        # 1. Vorwärts der Linie folgen (2,5 Meter)
        print("\n=== Schritt 1: Der Linie 2,5 Meter vorwärts folgen ===")
        if not follow_line_forward(FORWARD_DISTANCE, SPEED_FORWARD):
            print("Linienfolgen vorwärts konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 2. Drehen (180 Grad)
        print("\n=== Schritt 2: Drehen ===")
        if not turn(TURN_ANGLE, 20):
            print("Drehung konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # 3. Rückwärts der Linie folgen (zurück zum Ausgangspunkt)
        print("\n=== Schritt 3: Der Linie 2,5 Meter rückwärts folgen ===")
        if not follow_line_backward(FORWARD_DISTANCE, SPEED_BACKWARD):
            print("Linienfolgen rückwärts konnte nicht abgeschlossen werden.")
            return
        sleep(1)
        
        # Foto vom Endpunkt machen
        try:
            take_photo()
        except Exception as e:
            print(f"Hinweis: Foto konnte nicht aufgenommen werden: {e}")
        
        print("\n=== Autonomes Linienfolgen erfolgreich abgeschlossen! ===")
        
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
