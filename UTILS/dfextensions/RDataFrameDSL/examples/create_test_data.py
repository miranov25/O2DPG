#!/usr/bin/env python3
"""
Generate test ROOT files for RDataFrameDSL examples.

Creates:
- test_scalars.root: Simple scalar columns (px, py, pz, energy)
- test_tracks.root: RVec<TLorentzVector> tracks column

Usage:
    python create_test_data.py
"""

import ROOT
import numpy as np

def create_scalar_file(filename="test_scalars.root", n_events=1000):
    """Create ROOT file with scalar columns."""
    print(f"Creating {filename} with {n_events} events...")
    
    # Create TTree
    f = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("Events", "Scalar test data")
    
    # Variables
    px = np.array([0.0], dtype=np.float64)
    py = np.array([0.0], dtype=np.float64)
    pz = np.array([0.0], dtype=np.float64)
    energy = np.array([0.0], dtype=np.float64)
    
    # Branches
    tree.Branch("px", px, "px/D")
    tree.Branch("py", py, "py/D")
    tree.Branch("pz", pz, "pz/D")
    tree.Branch("energy", energy, "energy/D")
    
    # Fill with random data
    np.random.seed(42)
    for i in range(n_events):
        px[0] = np.random.normal(0, 10)
        py[0] = np.random.normal(0, 10)
        pz[0] = np.random.normal(0, 50)
        energy[0] = np.sqrt(px[0]**2 + py[0]**2 + pz[0]**2 + 0.14**2)  # pion mass
        tree.Fill()
    
    tree.Write()
    f.Close()
    print(f"  Created {n_events} events with px, py, pz, energy")


def create_tracks_file(filename="test_tracks.root", n_events=100):
    """Create ROOT file with RVec<TLorentzVector> tracks."""
    print(f"Creating {filename} with {n_events} events...")
    
    # Create TTree with RDataFrame (easier for RVec)
    # First create a simple tree, then add tracks via Define
    
    f = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("Events", "Track test data")
    
    # Use std::vector which RDataFrame converts to RVec
    tracks = ROOT.std.vector["TLorentzVector"]()
    tree.Branch("tracks", tracks)
    
    # Also add scalar pt array for comparison
    pt_vec = ROOT.std.vector["double"]()
    tree.Branch("pt", pt_vec)
    
    np.random.seed(42)
    for i in range(n_events):
        tracks.clear()
        pt_vec.clear()
        
        # Random number of tracks per event (1-10)
        n_tracks = np.random.randint(1, 11)
        
        for j in range(n_tracks):
            # Random kinematics
            pt = np.random.exponential(5.0)  # Exponential pT distribution
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            mass = 0.14  # pion mass
            
            # Create TLorentzVector
            tlv = ROOT.TLorentzVector()
            tlv.SetPtEtaPhiM(pt, eta, phi, mass)
            tracks.push_back(tlv)
            pt_vec.push_back(pt)
        
        tree.Fill()
    
    tree.Write()
    f.Close()
    print(f"  Created {n_events} events with 1-10 tracks each")


def create_particles_file(filename="test_particles.root", n_events=100):
    """Create ROOT file with RVec<TParticle> for property access demo."""
    print(f"Creating {filename} with {n_events} events...")
    
    f = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("Events", "Particle test data")
    
    # TParticle has public data members we can access
    particles = ROOT.std.vector["TParticle"]()
    tree.Branch("particles", particles)
    
    np.random.seed(42)
    pdg_codes = [211, -211, 321, -321, 2212, -2212]  # pions, kaons, protons
    
    for i in range(n_events):
        particles.clear()
        n_particles = np.random.randint(2, 8)
        
        for j in range(n_particles):
            pdg = int(np.random.choice(pdg_codes))  # Convert to Python int
            px = np.random.normal(0, 2)
            py = np.random.normal(0, 2)
            pz = np.random.normal(0, 5)
            mass = 0.14 if abs(pdg) == 211 else 0.49 if abs(pdg) == 321 else 0.94
            energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
            
            p = ROOT.TParticle()
            p.SetPdgCode(pdg)
            p.SetMomentum(px, py, pz, energy)
            particles.push_back(p)
        
        tree.Fill()
    
    tree.Write()
    f.Close()
    print(f"  Created {n_events} events with 2-7 particles each")


if __name__ == "__main__":
    print("=" * 50)
    print("RDataFrameDSL Test Data Generator")
    print("=" * 50)
    print()
    
    create_scalar_file()
    create_tracks_file()
    create_particles_file()
    
    print()
    print("Done! Files created:")
    print("  - test_scalars.root")
    print("  - test_tracks.root")
    print("  - test_particles.root")
    print()
    print("Run examples with:")
    print("  python 01_basic_usage.py")
    print("  python 03_method_broadcasting.py")
