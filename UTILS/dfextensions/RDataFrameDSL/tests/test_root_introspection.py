"""
Tests for ROOT Introspection Module

Phase: 13.6.D+
Purpose: Verify ROOT method discovery works correctly
"""

import pytest

# Skip if ROOT not available
try:
    import ROOT
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROOT_AVAILABLE,
    reason="ROOT not available"
)


# =============================================================================
# Test: Method Discovery for Known ROOT Classes
# =============================================================================

class TestRootClassDiscovery:
    """Test discovery on standard ROOT classes."""
    
    def test_discover_TLorentzVector(self):
        """Discover methods from TLorentzVector."""
        from RDataFrameDSL.root_introspection import discover_class_methods
        
        methods = discover_class_methods('TLorentzVector')
        
        # Should find key methods
        assert 'Pt' in methods, "Missing Pt() method"
        assert 'Eta' in methods, "Missing Eta() method"
        assert 'Phi' in methods, "Missing Phi() method"
        assert 'E' in methods, "Missing E() method"
        
        # Check return types
        assert methods['Pt'] == 'double', f"Pt() should return double, got {methods['Pt']}"
        assert methods['Eta'] == 'double'
        assert methods['Phi'] == 'double'
        assert methods['E'] == 'double'
        
        print(f"✅ TLorentzVector: Found {len(methods)} methods")
    
    def test_discover_TVector3(self):
        """Discover methods from TVector3."""
        from RDataFrameDSL.root_introspection import discover_class_methods
        
        methods = discover_class_methods('TVector3')
        
        # Should find vector methods
        assert 'X' in methods, "Missing X() method"
        assert 'Y' in methods, "Missing Y() method"
        assert 'Z' in methods, "Missing Z() method"
        assert 'Mag' in methods, "Missing Mag() method"
        
        print(f"✅ TVector3: Found {len(methods)} methods")
    
    def test_discover_TNamed(self):
        """Discover methods from TNamed."""
        from RDataFrameDSL.root_introspection import discover_class_methods
        
        methods = discover_class_methods('TNamed')
        
        # Should find name/title methods
        assert 'GetName' in methods, "Missing GetName() method"
        assert 'GetTitle' in methods, "Missing GetTitle() method"
        
        print(f"✅ TNamed: Found {len(methods)} methods")


# =============================================================================
# Test: Custom Class Discovery (Requires Declaration)
# =============================================================================

class TestCustomClassDiscovery:
    """Test discovery on custom classes."""
    
    @pytest.fixture(autouse=True)
    def setup_custom_classes(self):
        """Declare custom classes before tests."""
        ROOT.gInterpreter.Declare("""
        class TestCluster {
        public:
            double fQ;
            double fX;
            double fY;
            
            TestCluster() : fQ(0), fX(0), fY(0) {}
            TestCluster(double q, double x, double y) : fQ(q), fX(x), fY(y) {}
            
            double getQ() const { return fQ; }
            double getX() const { return fX; }
            double getY() const { return fY; }
            double r() const { return sqrt(fX*fX + fY*fY); }
        };
        
        class TestTrack {
        public:
            double fPx, fPy, fPz, fE;
            ROOT::RVec<TestCluster> fClusters;
            
            TestTrack() : fPx(0), fPy(0), fPz(0), fE(0) {}
            
            double Pt() const { return sqrt(fPx*fPx + fPy*fPy); }
            double Eta() const { return 0.5*log((Pt() + fPz)/(Pt() - fPz)); }
            double Phi() const { return atan2(fPy, fPx); }
            ROOT::RVec<TestCluster> clusters() const { return fClusters; }
            double totalCharge() const {
                double sum = 0;
                for (const auto& c : fClusters) sum += c.getQ();
                return sum;
            }
        };
        """)
        
        # Register pragmas
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class TestCluster+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class TestTrack+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class ROOT::VecOps::RVec<TestCluster>+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class ROOT::VecOps::RVec<TestTrack>+;')
    
    def test_discover_TestCluster(self):
        """Discover methods from TestCluster."""
        from RDataFrameDSL.root_introspection import discover_class_methods
        
        methods = discover_class_methods('TestCluster', verbose=True)
        
        # Should find our methods
        assert 'getQ' in methods, "Missing getQ() method"
        assert 'getX' in methods, "Missing getX() method"
        assert 'getY' in methods, "Missing getY() method"
        assert 'r' in methods, "Missing r() method"
        
        # Check return types
        assert methods['getQ'] == 'double'
        assert methods['getX'] == 'double'
        assert methods['getY'] == 'double'
        assert methods['r'] == 'double'
        
        print(f"✅ TestCluster: Found {len(methods)} methods")
        print(f"   Methods: {list(methods.keys())}")
    
    def test_discover_TestTrack(self):
        """Discover methods from TestTrack."""
        from RDataFrameDSL.root_introspection import discover_class_methods
        
        methods = discover_class_methods('TestTrack', verbose=True)
        
        # Should find our methods
        assert 'Pt' in methods, "Missing Pt() method"
        assert 'Eta' in methods, "Missing Eta() method"
        assert 'Phi' in methods, "Missing Phi() method"
        assert 'clusters' in methods, "Missing clusters() method"
        assert 'totalCharge' in methods, "Missing totalCharge() method"
        
        # Check return types
        assert methods['Pt'] == 'double'
        assert methods['clusters'] == 'RVec<TestCluster>', f"clusters() type: {methods['clusters']}"
        assert methods['totalCharge'] == 'double'
        
        print(f"✅ TestTrack: Found {len(methods)} methods")
        print(f"   Methods: {list(methods.keys())}")


# =============================================================================
# Test: Schema Generation
# =============================================================================

class TestSchemaGeneration:
    """Test automatic schema enhancement."""
    
    @pytest.fixture(autouse=True)
    def setup_custom_classes(self):
        """Declare custom classes."""
        # Reuse setup from previous test
        ROOT.gInterpreter.Declare("""
        #ifndef TESTCLUSTER_DEFINED
        #define TESTCLUSTER_DEFINED
        class TestCluster2 {
        public:
            double fQ;
            TestCluster2() : fQ(0) {}
            double getQ() const { return fQ; }
        };
        #endif
        
        #ifndef TESTTRACK_DEFINED
        #define TESTTRACK_DEFINED
        class TestTrack2 {
        public:
            double fPx, fPy;
            ROOT::RVec<TestCluster2> fClusters;
            
            TestTrack2() : fPx(0), fPy(0) {}
            double Pt() const { return sqrt(fPx*fPx + fPy*fPy); }
            ROOT::RVec<TestCluster2> clusters() const { return fClusters; }
        };
        #endif
        """)
        
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class TestCluster2+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class TestTrack2+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class ROOT::VecOps::RVec<TestCluster2>+;')
        ROOT.gInterpreter.ProcessLine('#pragma link C++ class ROOT::VecOps::RVec<TestTrack2>+;')
    
    def test_generate_schema_with_pragmas(self):
        """Test automatic schema enhancement."""
        from RDataFrameDSL.root_introspection import generate_schema_with_pragmas
        
        # Start with minimal schema
        original = {
            'event_id': 'long',
            'tracks': 'RVec<TestTrack2>',
        }
        
        # Auto-enhance
        enhanced = generate_schema_with_pragmas(original, verbose=True)
        
        # Should have pragmas
        assert '_pragmas' in enhanced, "Missing _pragmas key"
        pragmas = enhanced['_pragmas']
        
        # Should include TestTrack2 pragmas
        assert any('TestTrack2' in p for p in pragmas), "Missing TestTrack2 pragma"
        
        # Should have methods
        assert '_methods' in enhanced, "Missing _methods key"
        methods = enhanced['_methods']
        
        # Should have discovered TestTrack2 methods
        assert 'TestTrack2' in methods, "Missing TestTrack2 in _methods"
        assert 'Pt' in methods['TestTrack2'], "Missing Pt() in TestTrack2 methods"
        assert 'clusters' in methods['TestTrack2'], "Missing clusters() in TestTrack2 methods"
        
        print(f"✅ Schema enhanced:")
        print(f"   Pragmas: {len(pragmas)}")
        print(f"   Classes: {list(methods.keys())}")
        print(f"   TestTrack2 methods: {list(methods['TestTrack2'].keys())}")


# =============================================================================
# Test: Type Normalization
# =============================================================================

class TestTypeNormalization:
    """Test type string normalization."""
    
    def test_normalize_const_ref(self):
        """Test normalization of const references."""
        from RDataFrameDSL.root_introspection import normalize_type
        
        assert normalize_type('const double&') == 'double'
        assert normalize_type('const int&') == 'int'
        assert normalize_type('const TVector3&') == 'TVector3'
    
    def test_normalize_rvec(self):
        """Test normalization of ROOT::VecOps::RVec."""
        from RDataFrameDSL.root_introspection import normalize_type
        
        assert normalize_type('ROOT::VecOps::RVec<double>') == 'RVec<double>'
        assert normalize_type('ROOT::VecOps::RVec<int>') == 'RVec<int>'
        assert normalize_type('ROOT::VecOps::RVec<TestCluster>') == 'RVec<TestCluster>'
    
    def test_normalize_pointers(self):
        """Test normalization of pointers."""
        from RDataFrameDSL.root_introspection import normalize_type
        
        assert normalize_type('double*') == 'double'
        assert normalize_type('TObject*') == 'TObject'


# =============================================================================
# Test: Extract Custom Types
# =============================================================================

class TestExtractCustomTypes:
    """Test custom type extraction from schema."""
    
    def test_extract_simple(self):
        """Extract type from simple RVec."""
        from RDataFrameDSL.root_introspection import extract_custom_types
        
        types = extract_custom_types('RVec<ToyTrack>')
        assert types == ['ToyTrack']
    
    def test_extract_nested(self):
        """Extract type from nested RVec."""
        from RDataFrameDSL.root_introspection import extract_custom_types
        
        types = extract_custom_types('RVec<RVec<ToyCluster>>')
        assert types == ['ToyCluster']
    
    def test_extract_primitive(self):
        """Primitives should return empty list."""
        from RDataFrameDSL.root_introspection import extract_custom_types
        
        assert extract_custom_types('RVec<double>') == []
        assert extract_custom_types('RVec<int>') == []
        assert extract_custom_types('double') == []


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
