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

// we need dune-cornerpoint for reading the Dune grid.
#if HAVE_DUNE_CORNERPOINT
#include <dune/grid/CpGrid.hpp>
#else
#error This header needs the dune-cornerpoint module
#endif

#if HAVE_DUNE_ALUGRID
#include <dune/grid/common/datahandleif.hh>
#include <dune/alugrid/grid.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#else
#error This header needs the dune-alugrid module
#endif

#if HAVE_DUNE_FEM 
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/space/finitevolume.hh>
#endif

namespace Opm
{
    template <class GridImpl>     
    class DuneGrid
    {
    public:
        typedef GridImpl Grid;

        static const int dimension = Grid :: dimension ;

        typedef typename Grid :: Traits :: template Codim< 0 > :: Entity                 Element;
        typedef typename Grid :: Traits :: template Codim< 0 > :: EntityPointer          ElementPointer;
        typedef typename Grid :: Traits :: template Codim< dimension > :: Entity         Vertex;
        typedef typename Grid :: Traits :: template Codim< dimension > :: EntityPointer  VertexPointer;

        typedef Dune::Fem::AdaptiveLeafGridPart< Grid > GridPart;
        typedef typename GridPart :: GridViewType GridView;

        typedef typename Element :: Geometry ElementGeometry ;
        typedef typename ElementGeometry :: GlobalCoordinate GlobalCoordinate;

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

        Grid* createDuneGrid( Opm::DeckConstPtr deck, const std::vector<double>& porv )
        {
            if( porv.size() > 0 )
                OPM_THROW(std::runtime_error,"PORV not yet supported by DuneGrid");

            Dune::CpGrid cpgrid;
            // create CpGrid from deck
            cpgrid.processEclipseFormat(deck, 0.0, false, false);

            // grid factory converting a grid
            Dune::FromToGridFactory< Grid > factory;

            // create Grid from CpGrid
            std::vector< int > ordering;
            Grid& grid = *( factory.convert( cpgrid, ordering ) );

            for( int d=0; d<dimension; ++d )
                cartDims_[ d ] = cpgrid.logicalCartesianSize()[ d ];

            // compute cartesian dimensions 
            grid.comm().max( &cartDims_[ 0 ], dimension );

            globalIndex_.reset( new GlobalIndexContainer( grid, /* codim = */ 0 ) ); 
            globalIndex_->resize();

            GridPart gridPart( grid );

            // store global cartesian index of cell
            typedef typename GridPart :: template Codim< 0 > :: IteratorType Iterator;
            const Iterator end = gridPart.template end<0> ();
            int count = 0;
            for( Iterator it = gridPart.template begin<0> (); it != end; ++it, ++count )
            {
                const Element& element = *it;
                (*globalIndex_)[ element ] = cpgrid.globalCell()[ ordering[ count ] ];
            }

            // create data handle to distribute the global cartesian index
            DataHandle dh( *globalIndex_ );

            // partition grid
            grid.loadBalance( dh );
            // communicate non-interior cells values
            gridPart.communicate( dh, Dune::InteriorBorder_All_Interface, Dune::ForwardCommunication );
            // return grid pointer
            return &grid;
        }

        DuneGrid(Opm::DeckConstPtr deck, const std::vector<double>& porv )
            : grid_( createDuneGrid( deck, porv ) ),
              gridPart_( grid() ),
              //space_( gridPart_ )
              ug_( dune2UnstructuredGrid( gridPart_.gridView(), globalIndex(), cartDims_, true ) )
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

        const GlobalIndexContainer globalIndex() const { return *globalIndex_; }

        operator const UnstructuredGrid& () const { return *ug_; }
        operator UnstructuredGrid& () { return *ug_; }
        // return unstructured grid
        UnstructuredGrid & c_grid() { return *ug_; }
        const UnstructuredGrid& c_grid() const { return *ug_; }

        GridView gridView () const { 
            return gridPart_.gridView(); 
        }

        template <class GridView>
        UnstructuredGrid* 
        dune2UnstructuredGrid( const GridView& gridView, 
                               const GlobalIndexContainer& globalIndex,
                               const int cartDims[ dimension ],
                               const bool faceTags )
        {
            typedef typename Grid :: ctype ctype;
            typedef typename GridView :: template Codim< 0 > :: template Partition<
                Dune :: All_Partition > :: Iterator Iterator;
            typedef typename GridView :: IntersectionIterator            IntersectionIterator;
            typedef typename IntersectionIterator :: Intersection        Intersection;
            typedef typename Intersection :: Geometry                    IntersectionGeometry;
            typedef typename GridPart :: IndexSetType                    IndexSet;

            const IndexSet& indexSet = gridView.indexSet();

            const int numCells = indexSet.size( 0 );
            const int numFaces = indexSet.size( 1 );
            const int numNodes = indexSet.size( dimension );

            const int maxNumVerticesPerFace = 4;
            const int maxNumFacesPerCell    = 6;

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
            int maxFaceIdx = 0;
            const Iterator end = gridView.template end<0, Dune::All_Partition> ();
            for( Iterator it = gridView.template begin<0, Dune::All_Partition> (); it != end; ++it, ++count )
            {
                const Element& element = *it;
                const ElementGeometry geometry = element.geometry();

                // currently only hexahedrons are supported
                // assert( element.type().isHexahedron() );

                const int elIndex = indexSet.index( element );

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

                const Dune::ReferenceElement< ctype, dimension > &refElem
                        = Dune::ReferenceElements< ctype, dimension >::general( element.type() );

                int faceCount = 0;
                const IntersectionIterator endiit = gridView.iend( element );
                for( IntersectionIterator iit = gridView.ibegin( element ); iit != endiit; ++iit, ++faceCount )
                {
                    const Intersection& intersection = *iit;
                    IntersectionGeometry intersectionGeometry = intersection.geometry();
                    const double faceVol = intersectionGeometry.volume();

                    const int localFace = intersection.indexInInside();
                    const int localFaceIdx = isGhost ? 0 : localFace;
                    const int faceIndex = indexSet.subIndex( element, localFace, 1 );
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
        GridPart gridPart_;
        std::unique_ptr< UnstructuredGrid > ug_;
        int cartDims_[ dimension ];
    };
} // end namespace Opm
#endif
