/*
  Copyright 2014 IRIS AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_DUNEGRID_HEADER_INCLUDED
#define OPM_DUNEGRID_HEADER_INCLUDED

#include <opm/core/grid.h>
#include <opm/core/simulator/SimulatorState.hpp>
#include <opm/autodiff/DuneMatrix.hpp>

// we need dune-cornerpoint for reading the Dune grid.
#if HAVE_DUNE_CORNERPOINT
#include <dune/grid/CpGrid.hpp>
#else
#error This header needs the dune-cornerpoint module
#endif

#include <dune/grid/common/datahandleif.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#if HAVE_DUNE_ALUGRID
#include <dune/alugrid/grid.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#endif

#if HAVE_DUNE_FEM
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/space/finitevolume.hh>
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/function/blockvectorfunction.hh>
#include <dune/fem/function/combinedfunction.hh>
#include <dune/fem/operator/matrix/istlmatrixadapter.hh>
#include <dune/fem/operator/matrix/preconditionerwrapper.hh>
#endif

namespace Opm
{
    template <class DomainSpace, class RangeSpace = DomainSpace >
    class IstlMatrix : public DuneMatrix
    {
    public:
        typedef DuneMatrix  BaseType;
        typedef DomainSpace DomainSpaceType ;
        typedef RangeSpace  RangeSpaceType ;

#if HAVE_DUNE_FEM
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DomainSpaceType > RowDiscreteFunctionType ;
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< RangeSpaceType  > ColDiscreteFunctionType ;

        typedef typename RowDiscreteFunctionType :: DofStorageType  RowBlockVectorType;
        typedef typename ColDiscreteFunctionType :: DofStorageType  ColBlockVectorType;

        typedef typename RowDiscreteFunctionType :: GridType :: Traits ::
            CollectiveCommunication  CollectiveCommunictionType;
#endif

        IstlMatrix( const Eigen::SparseMatrix<double, Eigen::RowMajor>& matrix )
            : BaseType( matrix )
        {}
    };

    template <class GridImpl>
    class DuneGrid
    {
    public:
        typedef GridImpl Grid;
        typedef typename Grid :: CollectiveCommunication  CollectiveCommunication;

        static const int dimension = Grid :: dimension ;

        typedef typename Grid :: Traits :: template Codim< 0 > :: Entity                 Element;
        typedef typename Grid :: Traits :: template Codim< 0 > :: EntityPointer          ElementPointer;
        typedef typename Grid :: Traits :: template Codim< dimension > :: Entity         Vertex;
        typedef typename Grid :: Traits :: template Codim< dimension > :: EntityPointer  VertexPointer;

#if HAVE_DUNE_FEM
        typedef Dune::Fem::AdaptiveLeafGridPart< Grid >   AllGridPart;
        typedef typename AllGridPart :: GridViewType      AllGridView;
        typedef Dune::Fem::DGAdaptiveLeafGridPart< Grid > GridPart;
        typedef typename GridPart :: GridViewType         GridView;
#else
        typedef typename Grid :: LeafGridView             GridView;
        typedef GridView                                  AllGridView;
#endif

#if HAVE_DUNE_FEM
        typedef Dune::Fem::FunctionSpace< double, double, dimension, 1 >    FunctionSpace;
        typedef Dune::Fem::FiniteVolumeSpace< FunctionSpace, GridPart, /* codim = */ 0 >  FiniteVolumeSpace;
        typedef Dune::Fem::AdaptiveDiscreteFunction< FiniteVolumeSpace >    DiscreteFunction;

        typedef Dune::Fem::CombinedSpace< FiniteVolumeSpace, 3, Dune::Fem::VariableBased >  VectorSpaceType;

        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< FiniteVolumeSpace >  ISTLDiscreteFunction;
        typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< VectorSpaceType   >  ISTLVectorDiscreteFunction;

        // we can only deal with block size 1 at the moment
        static_assert( ISTLVectorDiscreteFunction::localBlockSize == 1, "blocksize error" );

        typedef IstlMatrix< VectorSpaceType   > SystemMatrixType;
        typedef IstlMatrix< FiniteVolumeSpace > EllipticMatrixType;

        typedef Dune::Fem::DGParallelMatrixAdapter< SystemMatrixType >   SystemMatrixAdapterType;
        typedef Dune::Fem::DGParallelMatrixAdapter< EllipticMatrixType > EllipticMatrixAdapterType;
#endif

        typedef typename Element :: Geometry                    ElementGeometry ;
        typedef typename ElementGeometry :: GlobalCoordinate    GlobalCoordinate;

        class GlobalCellIndex
        {
            int idx_;
        public:
            GlobalCellIndex() : idx_(-1) {}
            GlobalCellIndex& operator= ( const int index ) { idx_ = index; return *this; }
            int index() const { return idx_; }
        };

        typedef typename Dune::PersistentContainer< Grid, GlobalCellIndex > GlobalIndexContainer;

        class DataHandle : public Dune::CommDataHandleIF< DataHandle, int >
        {
            GlobalIndexContainer& globalIndex_;
        public:
            DataHandle( GlobalIndexContainer& globalIndex )
                : globalIndex_( globalIndex )
            {
                globalIndex_.resize();
            }

            bool contains ( int dim, int codim ) const { return codim == 0; }
            bool fixedsize( int dim, int codim ) const { return true; }

            //! \brief loop over all internal data handlers and call gather for
            //! given entity
            template<class MessageBufferImp, class EntityType>
            void gather (MessageBufferImp& buff, const EntityType& element ) const
            {
                int globalIdx = globalIndex_[ element ].index();
                buff.write( globalIdx );
            }

            //! \brief loop over all internal data handlers and call scatter for
            //! given entity
            template<class MessageBufferImp, class EntityType>
            void scatter (MessageBufferImp& buff, const EntityType& element, size_t n)
            {
                int globalIdx = -1;
                buff.read( globalIdx );
                if( globalIdx >= 0 )
                {
                    globalIndex_.resize();
                    globalIndex_[ element ] = globalIdx;
                }
            }

            //! \brief loop over all internal data handlers and return sum of data
            //! size of given entity
            template<class EntityType>
            size_t size (const EntityType& en) const
            {
                return 1;
            }
        };

        struct FaceKey : public std::pair< int, int >
        {
            typedef std::pair< int, int > BaseType;
            FaceKey() : BaseType(-1,-1) {}
            FaceKey( const int inside, const int outside )
                : BaseType( inside < outside ? std::make_pair(inside,outside) : std::make_pair(outside,inside) )
            {}
        };

        Grid* createDuneGrid( Opm::DeckConstPtr deck, const std::vector<double>& porv, const bool isCpGrid = false )
        {
            if( porv.size() > 0 )
                OPM_THROW(std::runtime_error,"PORV not yet supported by DuneGrid");

            std::unique_ptr< Dune::CpGrid > cpgrid;
            cpgrid.reset( new Dune::CpGrid() );

            // create CpGrid from deck
            cpgrid->processEclipseFormat(deck, 0.0, false, false);

            for( int d=0; d<dimension; ++d )
                cartDims_[ d ] = cpgrid->logicalCartesianSize()[ d ];

            // grid factory converting a grid
#if HAVE_DUNE_ALUGRID
            Dune::FromToGridFactory< Grid > factory;
#endif

            // store global cartesian index of cell
            std::map< int, int > globalIdMap ;
            int index = 0;
            /*
            for( auto it  = cpgrid.leafGridView().template begin<0>(),
                      end = cpgrid.leafGridView().template end<0>  (); it != end; ++it, ++index )
            {
                std::array<int,3> ijk;
                cpgrid.getIJK( index, ijk );
                const int globalId = ijk[ 0 ] + cartDims_[ 0 ] * ijk[ 1 ] + cartDims_[ 1 ] * cartDims_[ 0 ] * ijk[ 2 ];
                //const int globalId = ijk[ 1 ] + cartDims_[ 1 ] * ijk[ 2 ] + cartDims_[ 2 ] * cartDims_[ 1 ] * ijk[ 0 ];
                if( globalIdMap.find( globalId ) != globalIdMap.end() )
                    std::cout << "GlobalId not unique" << std::endl;
                //const int globalId = ijk[ 2 ] + cartDims_[ 2 ] * ijk[ 0 ] + cartDims_[ 0 ] * ijk[ 1 ];
                globalIdMap[ globalId ] = index;
            }
            */

            std::vector< int > ordering;
            ordering.reserve( globalIdMap.size() );

            index = 0;
            for( auto it = globalIdMap.begin(), end = globalIdMap.end(); it != end; ++it, ++index )
            {
                //std::cout << "ord[ " << index << " ] = " << (*it).second << std::endl;
                ordering.push_back( (*it).second );
            }

            // create Grid from CpGrid
#if HAVE_DUNE_ALUGRID
            Grid& grid = *( factory.convert( *cpgrid, ordering ) );
#else
            Grid& grid = *cpgrid;
#endif

            // compute cartesian dimensions
            grid.comm().max( &cartDims_[ 0 ], dimension );

            globalIndex_.reset( new GlobalIndexContainer( grid, /* codim = */ 0 ) );
            globalIndex_->resize();

#if HAVE_DUNE_FEM
            AllGridPart gridPart( grid );
            AllGridView gridView = gridPart.gridView();
#else
            AllGridView gridView = grid.leafGridView();
#endif

            // store global cartesian index of cell
            typedef typename AllGridView :: template Codim< 0 > :: Iterator Iterator;
            const Iterator end = gridView.template end<0> ();
            int count = 0;
            const bool orderingEmpty = ordering.empty();
            for( Iterator it = gridView.template begin<0> (); it != end; ++it, ++count )
            {
                const Element& element = *it;
                if( orderingEmpty )
                    (*globalIndex_)[ element ] = cpgrid->globalCell()[ count ];
                else
                    (*globalIndex_)[ element ] = cpgrid->globalCell()[ ordering[ count ] ];
            }

            // create data handle to distribute the global cartesian index
            DataHandle dh( *globalIndex_ );

            // partition grid
            grid.loadBalance( dh );
            // communicate non-interior cells values
            gridView.communicate( dh, Dune::InteriorBorder_All_Interface, Dune::ForwardCommunication );

#if ! HAVE_DUNE_ALUGRID
            cpgrid.release();
#endif
            return &grid;
        }

        DuneGrid(Opm::DeckConstPtr deck, const std::vector<double>& porv )
            : grid_( createDuneGrid( deck, porv ) ),
#if HAVE_DUNE_FEM
              allGridPart_( grid() ),
              gridPart_( grid() ),
              singleSpace_( gridPart_ ),
              vectorSpace_( gridPart_ ),
#endif
              ug_( dune2UnstructuredGrid(
#if HAVE_DUNE_FEM
                          allGridPart_.gridView(),
#else
                          grid().leafGridView(),
#endif
                          globalIndex(), cartDims_, true )
                 )
        {
            //printCurve( *grid_ );

            std::cout << "Created DuneGrid " << std::endl;
            std::cout << "P[ " << grid().comm().rank() << " ] = " << grid().size( 0 ) << std::endl;

            std::cout << "Created UG " << std::endl;
            std::cout << "P[ " << grid_->comm().rank() << " ] = " << ug_->number_of_cells << std::endl;
        }

        /// \brief destructor destroying the UnstructuredGrid
        ~DuneGrid()
        {
            // delete this structure manually
            UnstructuredGrid* ug = ug_.release();
            destroy_grid( ug );
        }

        // return Dune::Grid
        Grid& grid() { return *grid_; }
        const Grid& grid() const { return *grid_; }

#if HAVE_DUNE_FEM
        GridView gridView () const { return gridPart_.gridView(); }
#else
        GridView gridView () const { return grid().leafGridView(); }
#endif

        const GlobalIndexContainer globalIndex() const { return *globalIndex_; }

        // cast operators for UnstructuredGrid
        operator const UnstructuredGrid& () const { return *ug_; }
        operator       UnstructuredGrid& ()       { return *ug_; }

        // return unstructured grid
        UnstructuredGrid &      c_grid()       { return *ug_; }
        const UnstructuredGrid& c_grid() const { return *ug_; }

        const CollectiveCommunication& comm() const { return grid_->comm(); }

#if HAVE_DUNE_FEM
        void communicate( SimulatorState& state ) const
        {
            if( singleSpace_.size() != state.pressure().size()  )
                std::cout << singleSpace_.size() << " " << state.pressure().size() << std::endl;
            assert( singleSpace_.size() == state.pressure().size() );
            DiscreteFunction p( "pressure", singleSpace_, &state.pressure()[0] );
            p.communicate();
            DiscreteFunction sat( "sat", singleSpace_, &state.saturation()[0] );
            sat.communicate();
            DiscreteFunction sat2( "sat2", singleSpace_, &state.saturation()[ singleSpace_.size() ] );
            sat2.communicate();
        }

        SystemMatrixAdapterType matrixAdapter( SystemMatrixType& matrix ) {
            typedef Dune::Fem::FemSeqILU0< SystemMatrixType, typename SystemMatrixType::RowBlockVectorType,
                                                             typename SystemMatrixType::ColBlockVectorType > PreconditionerType;
            typename SystemMatrixAdapterType::PreconditionAdapterType
                precon( matrix, 0, 1.0, (PreconditionerType *) 0 );
            return SystemMatrixAdapterType( matrix, vectorSpace_, vectorSpace_, precon );
        }

        void gather( const SimulatorState& localState )
        {
            // gather solution to rank 0 for EclipseWriter
        }
#endif

        template <class GridView>
        UnstructuredGrid*
        dune2UnstructuredGrid( const GridView& gridView,
                               const GlobalIndexContainer& globalIndex,
                               const int cartDims[ dimension ],
                               const bool faceTags )
        {
            typedef double ctype;
            //typename Grid :: ctype ctype;
            typedef typename GridView :: template Codim< 0 > :: template Partition<
                Dune :: All_Partition > :: Iterator Iterator;
            typedef typename GridView :: IntersectionIterator      IntersectionIterator;
            typedef typename IntersectionIterator :: Intersection  Intersection;
            typedef typename Intersection :: Geometry              IntersectionGeometry;
            typedef typename GridView :: IndexSet                  IndexSet;

            const IndexSet& indexSet = gridView.indexSet();

            const int numCells = indexSet.size( 0 );
            const int numNodes = indexSet.size( dimension );

            const int maxNumVerticesPerFace = 4;
            const int maxNumFacesPerCell    = 6;

            int maxFaceIdx = indexSet.size( 1 );
            const bool validFaceIndexSet = maxFaceIdx > 0;

            std::map< FaceKey, int > faceIndexSet;

            if( ! validFaceIndexSet )
            {
                maxFaceIdx = 0;
                const Iterator end = gridView.template end<0, Dune::All_Partition> ();
                for( Iterator it = gridView.template begin<0, Dune::All_Partition> (); it != end; ++it )
                {
                    const Element& element = *it;
                    const int elIndex = indexSet.index( element );
                    const IntersectionIterator endiit = gridView.iend( element );
                    for( IntersectionIterator iit = gridView.ibegin( element ); iit != endiit; ++iit)
                    {
                        const Intersection& intersection = *iit;
                        int nbIndex = -1;
                        // store face --> cell relation
                        if( intersection.neighbor() )
                        {
                            ElementPointer ep = intersection.outside();
                            const Element& neighbor = *ep;
                            nbIndex = indexSet.index( neighbor );
                        }

                        FaceKey faceKey( elIndex, nbIndex );
                        if( faceIndexSet.find( faceKey ) == faceIndexSet.end() )
                            faceIndexSet[ faceKey ] = maxFaceIdx++;
                    }
                }
            }
            const int numFaces = maxFaceIdx ;

            // create Unstructured grid struct
            UnstructuredGrid* ug = allocate_grid( dimension, numCells, numFaces,
                    numFaces*maxNumVerticesPerFace,
                    numCells * maxNumFacesPerCell,
                    numNodes );

            std::fill( ug->face_cells, ug->face_cells+(numCells * maxNumFacesPerCell), -1 );

            for( int d=0; d<dimension; ++d )
              ug->cartdims[ d ] = cartDims_[ d ];

            assert( ug->number_of_cells > 0 );
            // allocate data structure for storage of cartesian index
            if( ! ug->global_cell )
                ug->global_cell = (int *) std::malloc( ug->number_of_cells * sizeof(int) );

            int count = 0;
            int cellFace = 0;
            maxFaceIdx = 0;
            const Iterator end = gridView.template end<0, Dune::All_Partition> ();
            for( Iterator it = gridView.template begin<0, Dune::All_Partition> (); it != end; ++it, ++count )
            {
                const Element& element = *it;
                const ElementGeometry geometry = element.geometry();

                // currently only hexahedrons are supported
                // assert( element.type().isHexahedron() );

                const int elIndex = indexSet.index( element );
                assert( indexSet.index( element ) == elIndex );

                const bool isGhost = element.partitionType() != Dune :: InteriorEntity ;

                // make sure that the elements are ordered as before,
                // otherwise the globalCell mapping is invalid
                assert( count == elIndex );

                // store cartesian index
                ug->global_cell[ elIndex ] = globalIndex[ element ].index();
                //std::cout << "global index of cell " << elIndex << " = " <<
                //    ug->global_cell[ elIndex ] << std::endl;

                const GlobalCoordinate center = geometry.center();
                int idx = elIndex * dimension;
                for( int d=0; d<dimension; ++d, ++idx )
                    ug->cell_centroids[ idx ] = center[ d ];
                ug->cell_volumes[ elIndex ] = geometry.volume();

                const int vertices = geometry.corners();
                for( int vx=0; vx<vertices; ++vx )
                {
                    const GlobalCoordinate vertex = geometry.corner( vx );
                    int idx = indexSet.subIndex( element, vx, dimension ) * dimension;
                    for( int d=0; d<dimension; ++d, ++idx )
                        ug->node_coordinates[ idx ] = vertex[ d ];
                }

                ug->cell_facepos[ elIndex ] = cellFace;

                Dune::GeometryType geomType = element.type();
                if( geomType.isNone() )
                    geomType = Dune::GeometryType( Dune::GeometryType::cube, dimension );

                const Dune::ReferenceElement< ctype, dimension > &refElem
                        = Dune::ReferenceElements< ctype, dimension >::general( geomType );

                int faceCount = 0;
                const IntersectionIterator endiit = gridView.iend( element );
                for( IntersectionIterator iit = gridView.ibegin( element ); iit != endiit; ++iit, ++faceCount )
                {
                    const Intersection& intersection = *iit;
                    IntersectionGeometry intersectionGeometry = intersection.geometry();
                    const double faceVol = intersectionGeometry.volume();

                    const int localFace = intersection.indexInInside();
                    const int localFaceIdx = isGhost ? 0 : localFace;

                    int faceIndex = validFaceIndexSet ? indexSet.subIndex( element, localFace, 1 ) : -1;
                    if( ! validFaceIndexSet )
                    {
                        int nbIndex = -1;
                        if( intersection.neighbor() )
                        {
                            ElementPointer ep = intersection.outside();
                            const Element& neighbor = *ep;

                            nbIndex = indexSet.index( neighbor );
                        }
                        FaceKey faceKey( elIndex, nbIndex );
                        faceIndex = faceIndexSet[ faceKey ];
                    }

                    maxFaceIdx = std::max( faceIndex, maxFaceIdx );

                    ug->face_areas[ faceIndex ] = faceVol;

                    // get number of vertices (should be 4)
                    const int vxSize = refElem.size( localFace, 1, dimension );
                    int faceIdx = faceIndex * maxNumVerticesPerFace ;
                    ug->face_nodepos[ faceIndex   ] = faceIdx;
                    ug->face_nodepos[ faceIndex+1 ] = faceIdx + maxNumVerticesPerFace;
                    for( int vx=0; vx<vxSize; ++vx, ++faceIdx )
                    {
                        const int localVx = refElem.subEntity( localFace, 1, vx, dimension );
                        const int vxIndex = indexSet.subIndex( element, localVx, dimension );
                        ug->face_nodes[ faceIdx ] = vxIndex ;
                    }

                    assert( vxSize    <= maxNumVerticesPerFace );
                    assert( localFace <  maxNumFacesPerCell );

                    // store cell --> face relation
                    ug->cell_faces  [ cellFace + localFaceIdx ] = faceIndex;
                    if( faceTags )
                    {
                        // fill logical cartesian orientation of the face (here indexInInside)
                        ug->cell_facetag[ cellFace + localFaceIdx ] = localFaceIdx;
                    }

                    GlobalCoordinate normal = intersection.centerUnitOuterNormal();
                    normal *= faceVol;

                    // store face --> cell relation
                    if( intersection.neighbor() )
                    {
                        ElementPointer ep = intersection.outside();
                        const Element& neighbor = *ep;

                        const int nbIndex = indexSet.index( neighbor );
                        if( elIndex < nbIndex )
                        {
                            ug->face_cells[ 2*faceIndex     ] = elIndex;
                            ug->face_cells[ 2*faceIndex + 1 ] = nbIndex;
                        }
                        else
                        {
                            ug->face_cells[ 2*faceIndex     ] = nbIndex;
                            ug->face_cells[ 2*faceIndex + 1 ] = elIndex;
                            // flip normal
                            normal *= -1.0;
                        }
                    }
                    else // domain boundary
                    {
                        ug->face_cells[ 2*faceIndex     ] = elIndex;
                        ug->face_cells[ 2*faceIndex + 1 ] = -1; // boundary
                    }

                    const GlobalCoordinate center = intersectionGeometry.center();
                    // store normal
                    int idx = faceIndex * dimension;
                    for( int d=0; d<dimension; ++d, ++idx )
                    {
                        ug->face_normals  [ idx ] = normal[ d ];
                        ug->face_centroids[ idx ] = center[ d ];
                    }
                }
                if( faceCount > maxNumFacesPerCell )
                    OPM_THROW(std::logic_error,"DuneGrid only supports conforming hexahedral currently");
                cellFace += faceCount;
            }

            // set last entry
            ug->cell_facepos[ numCells ] = cellFace;
            // set number of faces found
            ug->number_of_faces = maxFaceIdx+1;

            std::cout << cellFace << " " << indexSet.size( 1 ) << " " << maxFaceIdx << std::endl;

            return ug;
        }

    protected:
        std::unique_ptr< GlobalIndexContainer > globalIndex_;
        std::unique_ptr< Grid > grid_;
#if HAVE_DUNE_FEM
        AllGridPart allGridPart_;
        GridPart gridPart_;
        FiniteVolumeSpace singleSpace_;
        VectorSpaceType   vectorSpace_;
#endif
        std::unique_ptr< UnstructuredGrid > ug_;
        int cartDims_[ dimension ];
    };
} // end namespace Opm
#endif
