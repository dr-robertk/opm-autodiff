// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief A general-purpose simulator for ECL decks using the black-oil model.
 */
#include "config.h"

#include <opm/material/common/quad.hpp>
#include <ewoms/common/start.hh>
#include "eclproblem.hh"
#include <ewoms/disc/sofv/sofvdiscretization.hh>

BEGIN_PROPERTIES

NEW_TYPE_TAG(EclProblemSO, INHERITS_FROM(BlackOilModel, EclBaseProblem));
SET_TAG_PROP(EclProblemSO, SpatialDiscretizationSplice, SofvDiscretization);
// ebos can use a slightly faster stencil class because it does not need the normals and
// the integration points of intersections
SET_PROP(EclProblemSO, Stencil)
{
private:
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

public:
    typedef Ewoms::SofvStencil<TypeTag> type;
};

END_PROPERTIES

int main(int argc, char **argv)
{
    typedef TTAG(EclProblemSO) ProblemTypeTag;
    return Ewoms::start<ProblemTypeTag>(argc, argv);
}
