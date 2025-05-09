int External()
{
    int checkPdgSignal[] = {553,100553,200553};
    int checkPdgDecay = 13;
    double rapiditymin = -4.3; double rapiditymax = -2.3;
    std::string path{"o2sim_Kine.root"};
    std::cout << "Check for\nsignal PDG " << checkPdgSignal << "\ndecay PDG " << checkPdgDecay << "\n";
    TFile file(path.c_str(), "READ");
    if (file.IsZombie()) {
        std::cerr << "Cannot open ROOT file " << path << "\n";
        return 1;
    }

    auto tree = (TTree*)file.Get("o2sim");
    std::vector<o2::MCTrack>* tracks{};
    tree->SetBranchAddress("MCTrack", &tracks);

    int nLeptons{};
    int nAntileptons{};
    int nLeptonPairs{};
    int nLeptonPairsToBeDone{};
    int nSignalUpsilon1S{};
    int nSignalUpsilon2S{};
    int nSignalUpsilon3S{};
    int nSignalUpsilon1SWithinAcc{};
    int nSignalUpsilon2SWithinAcc{};
    int nSignalUpsilon3SWithinAcc{};
    auto nEvents = tree->GetEntries();
    o2::steer::MCKinematicsReader mcreader("o2sim", o2::steer::MCKinematicsReader::Mode::kMCKine);
    Bool_t isInjected = kFALSE;

    for (int i = 0; i < nEvents; i++) {
        tree->GetEntry(i);
        for (auto& track : *tracks) {
            auto pdg = track.GetPdgCode();
	        auto rapidity =  track.GetRapidity();
	        auto idMoth = track.getMotherTrackId();
            if (pdg == checkPdgDecay) {
                // count leptons
                nLeptons++;
            } else if(pdg == -checkPdgDecay) {
                // count anti-leptons
                nAntileptons++;
            } else if (pdg == checkPdgSignal[0] || pdg == checkPdgSignal[1] || pdg == checkPdgSignal[2]) {
                if(idMoth < 0){
                    // count signal PDG 
                    if (pdg == checkPdgSignal[0]) {
                        nSignalUpsilon1S++;
                    } else if (pdg == checkPdgSignal[1]) {
                        nSignalUpsilon2S++;
                    } else {
                        nSignalUpsilon3S++;
                    }
                    
                    // count signal PDG within acceptance 
                    if (rapidity > rapiditymin && rapidity < rapiditymax) {
                        if (pdg == checkPdgSignal[0]) {
                            nSignalUpsilon1SWithinAcc++;
                        } else if (pdg == checkPdgSignal[1]) {
                            nSignalUpsilon2SWithinAcc++;
                        } else {
                            nSignalUpsilon3SWithinAcc++;
                        }
                    }
		        }
		        auto child0 = o2::mcutils::MCTrackNavigator::getDaughter0(track, *tracks);
                auto child1 = o2::mcutils::MCTrackNavigator::getDaughter1(track, *tracks);
                if (child0 != nullptr && child1 != nullptr) {
                    // check for parent-child relations
                    auto pdg0 = child0->GetPdgCode();
                    auto pdg1 = child1->GetPdgCode();
                    std::cout << "First and last children of parent " << checkPdgSignal << " are PDG0: " << pdg0 << " PDG1: " << pdg1 << "\n";
                    if (std::abs(pdg0) == checkPdgDecay && std::abs(pdg1) == checkPdgDecay && pdg0 == -pdg1) {
                        nLeptonPairs++;
                        if (child0->getToBeDone() && child1->getToBeDone()) {
                            nLeptonPairsToBeDone++;
                        }
                    }
                }
            }
        }
    }
    std::cout << "#events: " << nEvents << "\n"
              << "#leptons: " << nLeptons << "\n"
              << "#antileptons: " << nAntileptons << "\n"
              << "#signal (Upsilon(1S)): " << nSignalUpsilon1S << "; within acceptance " << rapiditymin << " < y < " << rapiditymax << " : " << nSignalUpsilon1SWithinAcc << "\n"
              << "#signal (Upsilon(2S)): " << nSignalUpsilon2S << "; within acceptance " << rapiditymin << " < y < " << rapiditymax << " : " << nSignalUpsilon2SWithinAcc << "\n"
              << "#signal (Upsilon(2S)): " << nSignalUpsilon3S << "; within acceptance " << rapiditymin << " < y < " << rapiditymax << " : " << nSignalUpsilon3SWithinAcc << "\n"
              << "#lepton pairs: " << nLeptonPairs << "\n"
              << "#lepton pairs to be done: " << nLeptonPairs << "\n";


    if (nLeptonPairs == 0 || nLeptons == 0 || nAntileptons == 0) {
        std::cerr << "Number of leptons, number of anti-leptons as well as number of lepton pairs should all be greater than 1.\n";
        return 1;
    }
    if (nLeptonPairs != nLeptonPairsToBeDone) {
        std::cerr << "The number of lepton pairs should be the same as the number of lepton pairs which should be transported.\n";
        return 1;
    }

    return 0;
}