#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Let_picar_x_Move.py - Hauptmodul für die Steuerung des PiCar-X zwischen zwei roten Linien
"""

from picarx import Picarx
import time
from Linienverfolgung import ZweiLinienVerfolger

class LinienFahrt:
    def __init__(self):
        """Initialisiert die Linienfahrt"""
        # Picarx-Objekt erstellen
        self.px = Picarx()
        
        # Linienverfolger initialisieren
        self.linien_verfolger = ZweiLinienVerfolger(self.px)
        
        # Parameter für die Fahrt
        self.max_zeit = 180  # Maximale Zeit in Sekunden (3 Minuten)
        self.erste_strecke = 300  # Erste Strecke in mm (30 cm)
        self.zweite_strecke = 2000  # Zweite Strecke in mm (2 m)
        
        # Geschwindigkeiten
        self.fahr_geschwindigkeit = 25  # Normale Fahrgeschwindigkeit
        self.dreh_geschwindigkeit = 15  # Drehgeschwindigkeit
        self.park_geschwindigkeit = 15  # Parkgeschwindigkeit
        
        # Status-Variablen
        self.start_zeit = None
        self.ist_abgeschlossen = False
    
    def durchfuehren(self):
        """Führt die komplette Linienfahrt durch"""
        try:
            print("=== Starte Linienfahrt zwischen zwei roten Linien ===")
            
            # Startzeit merken
            self.start_zeit = time.time()
            
            # 1. Den beiden roten Linien für 30cm folgen
            print("\n=== Schritt 1: 30cm zwischen zwei roten Linien fahren ===")
            if not self.linien_verfolger.follow_lines_for_distance(self.erste_strecke, self.fahr_geschwindigkeit):
                print("Konnte 30cm nicht erfolgreich zurücklegen, breche ab")
                return False
            
            # Kurze Pause
            time.sleep(1)
            
            # 2. 180-Grad-Drehung zwischen den Linien
            print("\n=== Schritt 2: 180-Grad-Drehung zwischen den Linien ===")
            if not self.linien_verfolger.turn_180_between_lines(self.dreh_geschwindigkeit):
                print("180-Grad-Drehung fehlgeschlagen, breche ab")
                return False
            
            # Kurze Pause
            time.sleep(1)
            
            # 3. Den beiden roten Linien für 2m folgen
            print("\n=== Schritt 3: 2m zwischen zwei roten Linien fahren ===")
            if not self.linien_verfolger.follow_lines_for_distance(self.zweite_strecke, self.fahr_geschwindigkeit):
                print("Konnte 2m nicht erfolgreich zurücklegen, breche ab")
                return False
            
            # Kurze Pause
            time.sleep(1)
            
            # 4. Zwischen den Linien parken
            print("\n=== Schritt 4: Zwischen den roten Linien parken ===")
            if not self.linien_verfolger.park_between_lines(self.park_geschwindigkeit):
                print("Parken zwischen den Linien fehlgeschlagen, breche ab")
                return False
            
            # Gesamtzeit berechnen
            gesamt_zeit = time.time() - self.start_zeit
            
            print(f"\n=== Linienfahrt erfolgreich abgeschlossen in {gesamt_zeit:.1f} Sekunden ===")
            print("=== Fahrzeug steht im Parkplatz zwischen zwei roten Linien ===")
            
            self.ist_abgeschlossen = True
            return True
            
        except KeyboardInterrupt:
            print("\nProgramm durch Benutzer unterbrochen")
        except Exception as e:
            print(f"\nFehler bei der Linienfahrt: {e}")
        finally:
            # Aufräumen
            self.px.stop()
            try:
                self.linien_verfolger.cleanup()
            except:
                pass
            print("Programm beendet")
    
    def cleanup(self):
        """Aufräumen und Module schließen"""
        self.px.stop()
        try:
            self.linien_verfolger.cleanup()
        except:
            pass
        print("Linienfahrt beendet")

# Hauptprogramm
if __name__ == "__main__":
    try:
        # Linienfahrt initialisieren und durchführen
        linienfahrt = LinienFahrt()
        linienfahrt.durchfuehren()
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer unterbrochen")
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
    finally:
        # Aufräumen, falls noch nicht geschehen
        if 'linienfahrt' in locals():
            linienfahrt.cleanup()
