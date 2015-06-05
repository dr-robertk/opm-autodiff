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
#error "This header needs the dune-cornerpoint module"
#endif

#include <dune/grid/common/datahandleif.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#if HAVE_DUNE_ALUGRID
#include <dune/alugrid/grid.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#endif

#include <opm/autodiff/GridHelpers.hpp>

namespace Opm
{
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

        typedef typename Grid :: LeafGridView   GridView;

        typedef typename Element :: Geometry                    ElementGeometry ;
        typedef typename ElementGeometry :: GlobalCoordinate    GlobalCoordinate;

        struct CreateLeafGridView
        {
            GridView operator ()( Grid& grid ) const
            {
                return grid.leafGridView();
            }
        };


        // global id
        class GlobalCellIndex
        {
            int idx_;
        public:
            GlobalCellIndex() : idx_(-1) {}
            GlobalCellIndex& operator= ( const int index ) { idx_ = index; return *this; }
            int index() const { return idx_; }
        };

        typedef typename Dune::PersistentContainer< Grid, GlobalCellIndex > GlobalIndexContainer;

        // data handle for communicating global ids during load balance and communication
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

        // key for generating intersection index
        struct FaceKey : public std::pair< int, int >
        {
            typedef std::pair< int, int > BaseType;
            FaceKey() : BaseType(-1,-1) {}
            FaceKey( const int inside, const int outside )
                : BaseType( inside < outside ? std::make_pair(inside,outside) : std::make_pair(outside,inside) )
            {}
        };

        //! empty constructor
        explicit DuneGrid( std::pair< Grid*, GlobalIndexContainer* > gridAndIdx )
            : grid_( gridAndIdx.first ),
              globalIndex_( gridAndIdx.second ) {}

        //! constructor taking Eclipse deck and pore volumes
        DuneGrid(Opm::DeckConstPtr deck, const std::vector<double>& porv )
           // : grid_( createDuneGrid( deck, porv, CreateLeafGridView(), true ). ),
           //   ug_( dune2UnstructuredGrid( grid().leafGridView(), globalIndex(), cartDims_, true ) )
        {
            std::pair< Grid*, GlobalIndexContainer* >
                gridAndIdx( createDuneGrid( deck, porv, CreateLeafGridView(), true ) );
            grid_.reset( gridAndIdx.first );
            globalIndex_.reset( gridAndIdx.second );

            ug_.reset( dune2UnstructuredGrid( grid().leafGridView(), globalIndex(), cartDims_, true ) );

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

        //! underlying DUNE grid view
        GridView gridView () const { return grid().leafGridView(); }

        // cast operators for UnstructuredGrid
        operator const UnstructuredGrid& () const { return *ug_; }
        operator       UnstructuredGrid& ()       { return *ug_; }

        // return unstructured grid
        UnstructuredGrid &      c_grid()       { return *ug_; }
        const UnstructuredGrid& c_grid() const { return *ug_; }

        //! DUNE's collective communication object
        const CollectiveCommunication& comm() const { return grid_->comm(); }

    protected:
        // return global id container
        const GlobalIndexContainer globalIndex() const { assert( globalIndex_ ); return *globalIndex_; }

        // create the DUNE grid
        template <class CreateGridView>
        inline std::pair< Grid*, GlobalIndexContainer* > createDuneGrid( Opm::DeckConstPtr deck,
                                     const std::vector<double>& poreVolumes,
                                     const CreateGridView& createGridView,
                                     const bool distribute );

        // compute the global id for each cell
        template <class GV>
        inline GlobalIndexContainer* computeGlobalIndex( const GV& gridView,
                                                         Grid& grid,
                                                         const std::vector<int>& globalCell,
                                                         const std::vector<int>& ordering );

        // convert grid view into UnstructuredGrid struct
        template <class GV>
        inline UnstructuredGrid*
        dune2UnstructuredGrid( const GV& gridView,
                               const GlobalIndexContainer& globalIndex,
                               const int cartDims[ dimension ],
                               const bool faceTags );

        // compute the global id for each cell
        inline void distributeGrid( Grid& grid );

        // protected member variables
        std::unique_ptr< Grid > grid_;
        std::unique_ptr< GlobalIndexContainer > globalIndex_;
        std::unique_ptr< UnstructuredGrid > ug_;
        int cartDims_[ dimension ];
    };

    //////////////////////////////////////////////////////////////////////
    //
    //  Implementation
    //
    //////////////////////////////////////////////////////////////////////
    template <class GridImpl>
    template <class CreateGridView>
    inline std::pair< GridImpl*, typename DuneGrid<GridImpl>::GlobalIndexContainer*>
    DuneGrid<GridImpl>::createDuneGrid( Opm::DeckConstPtr deck,
                                        const std::vector<double>& poreVolumes,
                                        const CreateGridView& createGridView,
                                        const bool distribute )
    {
        std::unique_ptr< Dune::CpGrid > cpgrid;
        cpgrid.reset( new Dune::CpGrid() );

        // create CpGrid from deck
        cpgrid->processEclipseFormat(deck, false, false, false, poreVolumes );

        for( int d=0; d<dimension; ++d )
            cartDims_[ d ] = cpgrid->logicalCartesianSize()[ d ];

        std::vector< int > ordering;

        // grid factory converting a grid
        const std::vector< int >& globalCell = cpgrid->globalCell();

#if HAVE_DUNE_ALUGRID
        Dune::FromToGridFactory< Grid > factory;

        // store global cartesian index of cell
        /*
        std::map< int, int > globalIdMap ;
        int index = 0;
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

        /*
        ordering.reserve( globalIdMap.size() );

        index = 0;
        for( auto it = globalIdMap.begin(), end = globalIdMap.end(); it != end; ++it, ++index )
        {
            //std::cout << "ord[ " << index << " ] = " << (*it).second << std::endl;
            ordering.push_back( (*it).second );
        }
        */

        // create Grid from CpGrid
        Grid* grid = factory.convert( *cpgrid, ordering );
#else
        Grid* grid = cpgrid.release();
#endif
        auto gridView = createGridView( *grid );
        GlobalIndexContainer* globalIds = computeGlobalIndex( gridView, *grid, globalCell, ordering );

        if( distribute )
        {
            // distribute among all processes
            distributeGrid( *grid );
        }

        return std::make_pair( grid, globalIds );
    }


    template <class GridImpl>
    template <class GV>
    inline typename DuneGrid<GridImpl>::GlobalIndexContainer*
    DuneGrid<GridImpl>::computeGlobalIndex( const GV& gridView,
                                            Grid& grid,
                                            const std::vector<int>& globalCell,
                                            const std::vector<int>& ordering )
    {
        // compute cartesian dimensions for all cores
        grid.comm().max( &cartDims_[ 0 ], dimension );

        GlobalIndexContainer& globalIds = *(new GlobalIndexContainer( grid, /* codim = */ 0 ) );
        globalIds.resize();

        // store global cartesian index of cell
        typedef typename GV :: template Codim< 0 > :: Iterator Iterator;
        const Iterator end = gridView.template end<0> ();
        int count = 0;
        if( ordering.empty() )
        {
            for( Iterator it = gridView.template begin<0> (); it != end; ++it, ++count )
            {
                globalIds[ *it ] = globalCell[ count ];
            }
        }
        else
        {
            for( Iterator it = gridView.template begin<0> (); it != end; ++it, ++count )
            {
                assert( count < int(ordering.size()) );
                globalIds[ *it ] = globalCell[ ordering[ count ] ];
            }
        }

        return &globalIds;
    }


    template <class GridImpl>
    inline void
    DuneGrid<GridImpl>::distributeGrid ( Grid& grid )
    {
        assert( globalIndex_.operator ->() );

        // create data handle to distribute the global cartesian index
        DataHandle dh( *globalIndex_ );

        // partition grid
        grid.loadBalance( dh );

        // communicate non-interior cells values
        grid.communicate( dh, Dune::InteriorBorder_All_Interface, Dune::ForwardCommunication );
    }

    template <class GridImpl>
    template <class GV>
    inline UnstructuredGrid*
    DuneGrid<GridImpl>::dune2UnstructuredGrid( const GV& gridView,
                                               const GlobalIndexContainer& globalIndex,
                                               const int cartDims[ dimension ],
                                               const bool faceTags,
                                               const bool onlyInteriorCells )
    {
        typedef double ctype;
        //typename Grid :: ctype ctype;
        typedef typename GV :: template Codim< 0 > :: template Partition<
            Dune :: All_Partition > :: Iterator Iterator;
        typedef typename GV :: IntersectionIterator      IntersectionIterator;
        typedef typename IntersectionIterator :: Intersection  Intersection;
        typedef typename Intersection :: Geometry              IntersectionGeometry;
        typedef typename GV :: IndexSet                  IndexSet;

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
                        if( ! ( onlyInterior && neighbor.partitionType() != Dune::InteriorEntity ) )
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

                        if( ! ( onlyInterior && neighbor.partitionType() != Dune::InteriorEntity ) )
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

                    int nbIndex = -1;
                    if( ! ( onlyInterior && neighbor.partitionType() != Dune::InteriorEntity ) )
                        nbIndex = indexSet.index( neighbor );

                    if( nbIndex == -1 || elIndex < nbIndex )
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

        // std::cout << cellFace << " " << indexSet.size( 1 ) << " " << maxFaceIdx << std::endl;
        return ug;
    }

    namespace AutoDiffGrid
    {
        // derive from ADFaceCellTraits< UnstructuredGrid > since
        // DuneGrid casts into UnstructuredGrid
        template<class G>
        struct ADFaceCellTraits<DuneGrid< G > > : public ADFaceCellTraits< UnstructuredGrid >
        {
        };
    }

} // end namespace Opm

#endif
